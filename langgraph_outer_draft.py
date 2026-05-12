from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END

import logging

from langgraph_reasercher_draft import run_researcher_subgraph_node
from langgraph_writer_draft import run_writer_subgraph_node
from schema import ResearchPaperState


workflow = StateGraph(ResearchPaperState)

workflow.add_node("researcher", run_researcher_subgraph_node)
workflow.add_node("writer", run_writer_subgraph_node)

workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", "writer")
workflow.add_edge("writer", END)


checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)



def run_research_graph(research_topic: str, research_id: int) -> str:

    config: RunnableConfig = {
        "configurable": {"thread_id": f"manager_{research_id}"},
        "recursion_limit": 15
    }

    input_data = {"research_topic": research_topic, "research_id": research_id}

    logging.info("Запуск графа написания текста")

    result = app.invoke(input_data, config=config)

    final_pydantic_state = ResearchPaperState(**result)
    print(final_pydantic_state.raw_facts[:500])
    print("\n\n\n", "-="*10)
    return final_pydantic_state.main_paper_text

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    topic = input("Введите тему для исследования: ")
    result = run_research_graph(topic, research_id=1)
    print(result)
