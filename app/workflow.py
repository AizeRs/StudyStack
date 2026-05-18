from typing import Any, AsyncGenerator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

import logging

from app.config import settings
from app.agents.researcher import run_researcher_subgraph_node
from app.agents.writer import run_writer_subgraph_node
from app.agents.reviewer import run_macro_reviewer_subgraph_node
from app.schema import ResearchPaperState

def macro_reviewer_router(state: ResearchPaperState):
    if state.macro_reviewer_result == "approved":
        return "end"

    if state.macro_reviewer_result == "needs_facts":
        return "researcher"

    if state.macro_reviewer_iteration > settings.reviewers.macro_reviewer_max_iterations:
        # WIP: сохранение пометки о необходимости ручной проверки
        print("!!!"*10, "\nWARNING: FINAL TEXT VERSION DID NOT PASS THE REVIEWER!!!\n")
        print(f"FEEDBACK: {state.macro_reviewer_feedback}\n")
        return "end"

    scores_avg = sum(state.macro_reviewer_scores) / len(state.macro_reviewer_scores)

    if state.macro_reviewer_result == "needs_rewrite" and (
            scores_avg >= settings.reviewers.macro_reviewer_first_threshold) or (
        state.macro_reviewer_iteration > 2 and scores_avg >= settings.reviewers.macro_reviewer_second_threshold):
        return "end"

    return "writer"

workflow = StateGraph(ResearchPaperState)

workflow.add_node("researcher", run_researcher_subgraph_node)
workflow.add_node("writer", run_writer_subgraph_node)
workflow.add_node("macro_reviewer", run_macro_reviewer_subgraph_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", "macro_reviewer")

workflow.add_conditional_edges(
    "macro_reviewer",
    macro_reviewer_router,
    {
        "researcher": "researcher",
        "writer": "writer",
        "end": END
    }
)


checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)



async def run_research_graph(research_topic: str, research_id: str) -> str:

    config: RunnableConfig = {
        "configurable": {"thread_id": f"manager_{research_id}",
                         "macro_reviewer_max_iterations": settings.reviewers.macro_reviewer_max_iterations,
                         "researcher_recursion_limit": settings.researchers.recursion_limit},
        "recursion_limit": 50,
    }

    input_data = {"research_topic": research_topic, "research_id": research_id}

    logging.info("Запуск внешнего графа")

    result = await app.ainvoke(input_data, config=config)

    final_pydantic_state = ResearchPaperState(**result)
    print("\n\n\n", "-="*15 + "-")
    return final_pydantic_state.main_paper_text

async def stream_research_graph(research_topic: str, research_id: str) \
        -> AsyncGenerator[tuple[str, dict[str, Any]], None]:

    config: RunnableConfig = {
        "configurable": {"thread_id": f"manager_{research_id}",
                         "macro_reviewer_max_iterations": settings.reviewers.macro_reviewer_max_iterations,
                         "researcher_recursion_limit": settings.researchers.recursion_limit},
        "recursion_limit": 50,
    }

    input_data = {"research_topic": research_topic, "research_id": research_id}

    logging.info("Запуск внешнего графа")

    async for event in app.astream(input_data, config=config, stream_mode="updates"):
        for node_name, state_update in event.items():
            logging.info(f"Узел '{node_name}' завершил работу.")

            yield node_name, state_update
