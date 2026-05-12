from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from config import llm

import logging
from schema import ResearchPaperState, ReviewerState


SYSTEM_PROMPT_REVIEWER = """"""


def agent_node(state: ReviewerState) -> dict:
    response = llm.invoke(state.messages)

    return {"messages": [response]}


workflow = StateGraph(ReviewerState)

workflow.add_node("agent", agent_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)


def run_writer_subgraph_node(state: ResearchPaperState) -> dict:

    config: RunnableConfig = {
        "configurable": {"thread_id": f"writer_{state.research_id}"},
        "recursion_limit": 15
    }


    if state.feedback and state.review_result_first == "needs_rewrite":
        input_data = {"messages": [HumanMessage(content=f"Критик забраковал твой текст. Замечание: {state.feedback}. Исправь!")]}
    else:

        input_data = {
            "messages": [
                SystemMessage(content=SYSTEM_PROMPT_WRITER),
                HumanMessage(content="Материалы из ресерча: " + state.raw_facts),
                HumanMessage(content=f"Тема отчёта: {state.research_topic}. Начинай писать его сейчас.")]}

    logging.info("Инференс писателя текста")

    result = app.invoke(input_data, config=config)

    final_pydantic_state = WriterState(**result)
    return {"main_paper_text": final_pydantic_state.messages[-1].content}
