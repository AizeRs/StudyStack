from langgraph.graph import StateGraph, START, END
import logging

from app.schema import ChapterSubgraphState
from app.agents.writer import run_writer_node
from app.agents.reviewer import run_reviewer_node
from app.agents.linter import run_linter_node

def reviewer_router(state: ChapterSubgraphState):
    if state.reviewer_result == "approved":
        return "linter"
    
    if state.iteration > 3:
        logging.warning(f"[ChapterSubgraph] Глава {state.chapter_index} не прошла проверку за 3 итерации. Пропускаем в линтер как есть.")
        return "linter"
        
    return "writer"

chapter_workflow = StateGraph(ChapterSubgraphState)

chapter_workflow.add_node("writer", run_writer_node)
chapter_workflow.add_node("reviewer", run_reviewer_node)
chapter_workflow.add_node("linter", run_linter_node)

chapter_workflow.add_edge(START, "writer")
chapter_workflow.add_edge("writer", "reviewer")
chapter_workflow.add_conditional_edges("reviewer", reviewer_router)
chapter_workflow.add_edge("linter", END)

chapter_subgraph_app = chapter_workflow.compile()
