"""
Основной рабочий процесс (Workflow) генерации исследования.
Описывает структуру графа LangGraph, маршрутизацию между узлами
и логику параллельного выполнения (fan-out/fan-in) для анализа источников и написания глав.
"""
from typing import AsyncGenerator

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

import logging

from app.config import settings
from app.agents.planner import macro_planner_node
from app.agents.source_finder import run_researcher_subgraph_node
from app.agents.source_analyzer import run_source_analyzer_node
from app.agents.scriptor import run_scriptor_node
from app.agents.chapter_subgraph import chapter_subgraph_app
from app.schema import ResearchPaperState


# ── Fan-out: параллельный запуск Source Analyzer по каждому источнику ──

def fan_out_sources(state: ResearchPaperState):
    # Для каждой пары (doc_id, url) создаёт Send в узел source_analyzer.
    # Результаты сливаются через reducer mapped_data.
    if not state.source_registry:
        logging.warning("Source registry пуст — source_finder не нашёл ни одного источника.")
        return "scriptor"

    plan_strings = [section.name for section in state.macro_plan]

    return [
        Send("source_analyzer", {
            "doc_id": doc_id,
            "url": url,
            "research_paper_plan": plan_strings,
        })
        for doc_id, url in state.source_registry.items()
    ]


# ── Fan-out: параллельный запуск Chapter Subgraph по каждой главе ──

async def run_chapter_subgraph_node(state: dict, config: RunnableConfig):
    # Обертка для запуска субграфа главы.
    # Возвращает только ключи, присутствующие в родительском state.
    result = await chapter_subgraph_app.ainvoke(state, config=config)
    # Возвращаем только сгенерированный текст главы для редюсера
    if "chapter_drafts" in result:
        return {"chapter_drafts": result["chapter_drafts"]}
    return {}

def fan_out_chapters(state: ResearchPaperState):
    # Для каждой главы из macro_plan создаёт Send в chapter_subgraph.
    # Результаты собираются в chapter_drafts.
    sends = []
    num_chapters = len(state.macro_plan)
    biblio_idx = num_chapters - 1 if num_chapters > 0 else -1

    for i, section in enumerate(state.macro_plan):
        skeleton = state.chapter_skeletons.get(i, {})
        
        if i == biblio_idx:
            facts = list(state.source_registry.values())
        else:
            facts = state.mapped_data.get(i, [])
            
        sends.append(Send("chapter_subgraph", {
            "chapter_index": i,
            "chapter_name": section.name,
            "core_thesis": skeleton.get("core_thesis", ""),
            "bridge_to_next": skeleton.get("bridge_to_next", ""),
            "facts": facts,
            "is_bibliography": i == biblio_idx,
            "is_intro_or_conclusion": (i == 0) or (i == biblio_idx - 1),
            "page_count": section.page_count,
            "academic_level": state.academic_level,
            "additional_instructions": state.additional_instructions
        }))
    return sends


def final_concatenation_node(state: ResearchPaperState):
    # Собирает готовые и отформатированные тексты глав в итоговый документ.
    return {"main_paper_text": final_text}


# ── Граф ──

workflow = StateGraph(ResearchPaperState)

workflow.add_node("planner", macro_planner_node)
workflow.add_node("source_finder", run_researcher_subgraph_node)
workflow.add_node("source_analyzer", run_source_analyzer_node)
workflow.add_node("scriptor", run_scriptor_node)
workflow.add_node("chapter_subgraph", run_chapter_subgraph_node)
workflow.add_node("final_concatenator", final_concatenation_node)

# START → планировщик → поиск источников
workflow.add_edge(START, "planner")
workflow.add_edge("planner", "source_finder")

# поиск → параллельный анализ (fan-out через Send)
workflow.add_conditional_edges("source_finder", fan_out_sources)

# fan-in → Сценарист (создаёт скелеты глав)
workflow.add_edge("source_analyzer", "scriptor")

# Сценарист → параллельные писатели (fan-out через Send)
workflow.add_conditional_edges("scriptor", fan_out_chapters)

# fan-in → финальная склейка текста
workflow.add_edge("chapter_subgraph", "final_concatenator")
workflow.add_edge("final_concatenator", END)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)


# ── Точки входа (используются ботом и main.py) ──

async def run_research_graph(research_topic: str, research_id: str, research_length: int = 3, academic_level: str = "college", additional_instructions: str = "") -> str:

    config: RunnableConfig = {
        "configurable": {
            "thread_id": f"manager_{research_id}",
            "researcher_recursion_limit": settings.researchers.recursion_limit,
        },
        "recursion_limit": 50,
    }

    input_data = {
        "research_topic": research_topic,
        "research_id": research_id,
        "research_length": research_length,
        "academic_level": academic_level,
        "additional_instructions": additional_instructions
    }

    logging.info("Запуск внешнего графа")

    result = await app.ainvoke(input_data, config=config)

    final_pydantic_state = ResearchPaperState(**result)
    return final_pydantic_state.main_paper_text


async def stream_research_graph(research_topic: str, research_id: str, research_length: int = 3, academic_level: str = "college", additional_instructions: str = "") \
        -> AsyncGenerator[dict, None]:
    import asyncio
    
    progress_queue = asyncio.Queue()

    config: RunnableConfig = {
        "configurable": {
            "thread_id": f"manager_{research_id}",
            "researcher_recursion_limit": settings.researchers.recursion_limit,
            "progress_queue": progress_queue
        },
        "recursion_limit": 50,
    }

    input_data = {
        "research_topic": research_topic,
        "research_id": research_id,
        "research_length": research_length,
        "academic_level": academic_level,
        "additional_instructions": additional_instructions
    }

    logging.info("Запуск внешнего графа (стриминг)")

    async def run_graph():
        try:
            async for event in app.astream(input_data, config=config, stream_mode="updates"):
                for node_name, state_update in event.items():
                    logging.info(f"Узел '{node_name}' завершил работу.")
                    await progress_queue.put({"type": "node_update", "node_name": node_name, "state_update": state_update})
        except Exception as e:
            logging.error(f"Ошибка в графе: {e}", exc_info=True)
            await progress_queue.put({"type": "error", "message": str(e)})
        finally:
            await progress_queue.put(None)
            
    task = asyncio.create_task(run_graph())
    
    while True:
        event = await progress_queue.get()
        if event is None:
            break
        yield event
    
    await task
