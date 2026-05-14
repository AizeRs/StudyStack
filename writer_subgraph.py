from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from config import llm

import logging
from schema import WriterState, ResearchPaperState


SYSTEM_PROMPT_WRITER = """
Role: Expert Content Architect.
Mission: Transform the provided research dossier into a polished, high-impact narrative.
Core Rules:
Fidelity: Use only the provided data and quotes. Do not invent facts.
Synthesis: Weave raw evidence into a seamless, logical flow (not a list).
Adaptation: Strictly follow the user’s requested tone, length, and complexity.
Polishing: Eliminate filler and ensure professional, engaging transitions.
Invisible Process: The final text must be a standalone piece with no mention of the "researcher" or "dossier."
Output: A ready-to-publish document based exclusively on the provided evidence.
"""


def agent_node(state: WriterState) -> dict:
    response = llm.invoke(state.messages)

    return {"messages": [response]}


workflow = StateGraph(WriterState)

workflow.add_node("agent", agent_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)


def run_writer_subgraph_node(state: ResearchPaperState) -> dict:

    config: RunnableConfig = {
        "configurable": {"thread_id": f"writer_{state.research_id}_v{state.facts_version}"},
    }


    if state.macro_reviewer_feedback and state.macro_reviewer_result == "needs_rewrite":
        input_data = {"messages": [HumanMessage(content=f"Критик забраковал твой текст. Замечание: {state.macro_reviewer_feedback}. Исправь!")]}
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
