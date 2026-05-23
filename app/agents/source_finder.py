from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import asyncio

from app.config import llm
from app.tools.search import search_web
from langgraph.errors import GraphRecursionError

import logging
from app.schema import SourceFinderState, ResearchPaperState, FinalSources


SYSTEM_PROMPT_SOURCE_FINDER = """Role: Expert Research Librarian & Source Locator.
Mission: Your sole objective is to discover, evaluate, and collect exactly {target_source_count} highly relevant, credible, and diverse web sources for the given research topic.\n
Core Workflow & Rules:
1. Tool Usage: You must use the `search_web` tool to explore the topic. The tool will return a list of search results, each containing a Title, a brief Snippet, and a unique `doc_id`.
2. Evaluation: Carefully analyze the provided snippets. Select only the most informative and authoritative sources that provide a strong foundation for an academic or professional paper.
3. Iterative Search: If the initial search results are poor, or if you haven't found enough high-quality sources, you must reformulate your query and use the `search_web` tool again. 
4. Target Quota: You MUST continue searching until you have identified exactly {target_source_count} optimal sources. 
5. Strict Boundaries: You are NOT a writer or a synthesizer. Do not attempt to write the final report, summarize the topic extensively, or answer the user's prompt directly. 
6. Zero Hallucination: Never invent or modify a `doc_id`. You may only select `doc_id`s explicitly returned to you by the `search_web` tool.\n
Completion Trigger:
Once you have successfully gathered exactly {target_source_count} relevant `doc_id`s, output a final message clearly listing these selected `doc_id`s along with a very brief justification (1 sentence max) for each, so the downstream system can process them."""



my_tools = [search_web]
tools_by_name = {t.name: t for t in my_tools}


async def custom_tools_node(state: SourceFinderState, config: RunnableConfig = None) -> dict:
    last_message = state.messages[-1]
    
    progress_queue = config.get("configurable", {}).get("progress_queue") if config else None

    if not getattr(last_message, 'tool_calls', None):
        return {"messages": []}

    tasks = []
    for call in last_message.tool_calls:
        if progress_queue and call["name"] == "search_web":
            query = call["args"].get("query", "...")
            await progress_queue.put({"type": "custom", "message": f"🔍 Гуглю: {query}... ⏳"})
            
        tool = tools_by_name.get(call["name"])
        if tool:
            tasks.append(tool.ainvoke(call))

    if not tasks:
        return {"messages": []}

    # Последовательный вызов инструментов с задержкой (защита от 429 Too Many Requests)
    tool_messages = []
    for task in tasks:
        result = await task
        tool_messages.append(result)
        await asyncio.sleep(2)

    messages_to_add = []
    new_urls = {}

    for tool_msg in tool_messages:
        messages_to_add.append(tool_msg)

        if hasattr(tool_msg, 'artifact') and isinstance(tool_msg.artifact, dict):
            new_urls.update(tool_msg.artifact)

    return {
        "messages": messages_to_add,
        "url_registry": new_urls
    }

async def agent_node(state: SourceFinderState) -> dict:
    llm_with_tools = llm.bind_tools(my_tools)
    response = await llm_with_tools.ainvoke(state.messages)

    return {"messages": [response]}

def should_continue(state: SourceFinderState) -> str:
    last_message = state.messages[-1]

    if getattr(last_message, 'tool_calls', None):
        logging.info("Модель вызвала тул, передаём управление ему")
        return "continue"

    logging.info("Модель выдала ответ, завершаем цикл")
    return "end"

async def formatter_node(state: SourceFinderState) -> dict:
    logging.info("Поиск завершен. Форматируем финальный вывод...")

    structured_llm = llm.with_structured_output(FinalSources, method="function_calling")

    extraction_prompt = HumanMessage(
        content="Поиск завершен. Проанализируй контекст выше и извлеки все релевантные doc_id, "
                "которые ты решил использовать для ответа. Верни их строгим массивом."
    )

    response: FinalSources = await structured_llm.ainvoke(state.messages + [extraction_prompt])

    return {
        "final_doc_ids": response.doc_ids
    }


workflow = StateGraph(SourceFinderState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", custom_tools_node)
workflow.add_node("formatter", formatter_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": "formatter"
    }
)

workflow.add_edge("tools", "agent")
workflow.add_edge("formatter", END)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)


async def run_researcher_subgraph_node(state: ResearchPaperState, config: RunnableConfig = None) -> dict:
    session_id = uuid4().hex[:8]
    
    progress_queue = config.get("configurable", {}).get("progress_queue") if config else None

    researcher_config: RunnableConfig = {
        "configurable": {
            "thread_id": f"researcher_{state.research_id}_{session_id}",
            "progress_queue": progress_queue
        },
        "recursion_limit": config.get("configurable", {}).get("researcher_recursion_limit", 20) if config else 20,
    }

    topic = state.research_topic

    target_count = state.research_length + 7
    formatted_system_prompt = SYSTEM_PROMPT_SOURCE_FINDER.format(target_source_count=target_count)

    input_data = {
        "messages": [
            SystemMessage(content=formatted_system_prompt),
            HumanMessage(content=f"TOPIC: {topic}. Start searching for sources now.")
        ]
    }

    try:
        logging.info("Инференс сборщика данных")

        result = await app.ainvoke(input_data, config=researcher_config)
        final_state = SourceFinderState(**result)

    except GraphRecursionError:
        logging.info("Достигнут лимит вызовов тулов. Вызываем formatter для graceful shutdown.")

        crash_state = await app.aget_state(researcher_config)
        state_values = crash_state.values

        messages = state_values.get("messages", [])

        # Удаление незавершенного вызова инструмента (AIMessage), вызвавшего ошибку лимита
        if messages and isinstance(messages[-1], AIMessage) and getattr(messages[-1], 'tool_calls', None):
            logging.info("Удаляем незавершенный ответ модели перед форматированием.")
            messages.pop(-1)
            state_values["messages"] = messages

        current_state = SourceFinderState(**state_values)

        # Форматирование ответа после прерывания
        formatter_result = await formatter_node(current_state)

        # Имитация успешного завершения
        current_state.final_doc_ids = formatter_result.get("final_doc_ids", [])
        final_state = current_state


    collected_ids = final_state.final_doc_ids
    url_registry = final_state.url_registry

    # Подстановка URL по doc_id
    final_sources = {}
    for doc_id in collected_ids:
        if doc_id in url_registry:
            final_sources[doc_id] = url_registry[doc_id]
        else:
            logging.warning(f"Модель сгаллюцинировала doc_id: {doc_id} (не найден в url_registry). Пропускаем.")


    return {"source_registry": final_sources}
