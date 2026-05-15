from uuid import uuid4

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, SystemMessage
from langgraph.prebuilt import ToolNode

from app.config import llm
from app.tools.search import search_web, read_webpage
from langgraph.errors import GraphRecursionError

import logging
from app.schema import ResearcherState, ResearchPaperState


SYSTEM_PROMPT_RESEARCHER = """
Role: You are a Senior Research Specialist. Your mission is to conduct exhaustive investigations and gather high-quality "raw material" for a professional writer. You do not write the final narrative; instead, you provide the structural foundation and evidence required to create a masterpiece.
Available Tools:
search_web: Use this to identify key sources, news, academic papers, and diverse perspectives.
read_webpage: Use this to extract detailed information, data points, and direct quotes from specific URLs.
Core Objectives:
Depth over Breadth: Don't just skim the surface. Find the nuances, contradictions, and specific details that a general search would miss.
Direct Evidence: You must provide direct, verbatim quotes from credible sources.
Source Integrity: Always attribute information to a specific URL and title.
Data Categorization: Organize the gathered material into logical "buckets" (e.g., Historical Context, Key Stakeholders, Technical Specifications, Current Controversies).
Research Methodology & Workflow
Phase 1: Initial Discovery. Perform multiple search_web queries using different keywords to ensure a 360-degree view of the topic.
Phase 2: Deep Extraction. Use read_webpage on at least 2 high-quality sources. Look for specific statistics, expert opinions.
Phase 3: Synthesis. Group the information into a structured research dossier.
Output Format Requirements
Your final output must be a comprehensive document structured as follows:
Executive Summary of Findings: A brief overview of what the research uncovered.
Key Themes & Data Points:
Theme Title:
Detailed findings.
Direct Quotes: "Quote text here" — [Source Name/URL].
Data/Statistics: Hard numbers and facts.
Primary Sources & Citations: A list of all URLs visited with a one-sentence description of what each provides.
Suggested Angles for the Writer: Ideas on how the writer could frame this story based on the evidence found.
Operational Constraints
No Hallucinations: If you cannot find a specific fact, state that it was not found.
Verbatim Accuracy: Quotes must be 100% accurate to the source text.
Tone: Objective, analytical, and professional.
"""

my_tools = [search_web, read_webpage]
tool_node = ToolNode(my_tools)


def agent_node(state: ResearcherState) -> dict:
    llm_with_tools = llm.bind_tools(my_tools)
    response = llm_with_tools.invoke(state.messages)

    return {"messages": [response]}


def should_continue(state: ResearcherState) -> str:
    last_message = state.messages[-1]

    if getattr(last_message, 'tool_calls', None):
        logging.info("Модель вызвала тул, передаём управление ему")
        return "continue"

    logging.info("Модель выдала ответ, завершаем цикл")
    return "end"


workflow = StateGraph(ResearcherState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")

workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END
    }
)

workflow.add_edge("tools", "agent")

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)



def run_researcher_subgraph_node(state: ResearchPaperState, config: RunnableConfig) -> dict:

    session_id = uuid4().hex[:8]

    researcher_config: RunnableConfig = {
        "configurable": {"thread_id": f"researcher_{state.research_id}_{session_id}"},
        "recursion_limit": config.get("configurable", {}).get("researcher_recursion_limit", 20),
    }

    task = state.research_topic
    is_incremental_search = state.macro_reviewer_feedback and state.macro_reviewer_result == "needs_facts"

    if is_incremental_search:
        delta_prompt = f"""
            Original Topic: {task}
            Current Research: {state.raw_facts}
            The writer is trying to draft the paper, but the Reviewer noted missing crucial information.
            Reviewer Feedback: {state.macro_reviewer_feedback}
            IMPORTANT: We already have the foundational data. 
            Your job is ONLY to find the specific missing facts requested by the reviewer. 
            Do not do a general search. Do not send the text of the current research again. 
            Only send the extension of it - missing specific facts which you were asked for.
            """

        input_data = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT_RESEARCHER),
                HumanMessage(content=delta_prompt)
            ]
        }

    else:
        input_data = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT_RESEARCHER),
                HumanMessage(content=SYSTEM_PROMPT_RESEARCHER + f"TASK: {task}. Start your research now.")
            ]}

    try:
        logging.info("Инференс сборщика данных")

        result = app.invoke(input_data, config=researcher_config)

        final_pydantic_state = ResearcherState(**result)
        if is_incremental_search:
            return {"raw_facts": state.raw_facts + "\n\n--- ADDITIONAL RESEARCH DATA (PARTIAL) ---\n\n"
                                 + final_pydantic_state.messages[-1].content,
                    "extra_research_forbidden": True,
                    "facts_version": state.facts_version + 1,
                    "macro_reviewer_iteration": 1}
        return {"raw_facts": final_pydantic_state.messages[-1].content}

    except GraphRecursionError:
        logging.info("Достигнут лимит вызовов тулов. Начинаем graceful shutdown")

        crash_state = app.get_state(researcher_config)

        messages = crash_state.values.get("messages", [])

        if messages and isinstance(messages[-1], AIMessage):
            logging.info("Цикл оборвался до результата выполнения тула. Удаляем последний ответ модели и добавляем системное уведомление.")
            messages.pop(-1)

        if messages and isinstance(messages[-1], ToolMessage):
            messages[-1].content += "\n\n[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Лимит поиска исчерпан. Сформируй итоговый ответ на основе этих данных."

            final_result = llm.invoke(messages)

            if is_incremental_search:
                return {"raw_facts": state.raw_facts + "\n\n--- ADDITIONAL RESEARCH DATA (PARTIAL) ---\n\n" +
                                     final_result.content,
                        "extra_research_forbidden": True,
                        "facts_version": state.facts_version + 1,
                        "macro_reviewer_iteration": 1}

            return {"raw_facts": final_result.content}

        else:
            print(f"Ошибка! Невозможно добавить системное уведомление. {len(messages)=} {type(messages[-1])=}")
            raise
