"""
Определение структур данных, схем Pydantic и состояний (State) для LangGraph.
Содержит функции-редюсеры (reducers) для агрегации данных при параллельном выполнении узлов.
"""
from typing import Annotated, Optional, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field




def merge_registries(left: dict, right: dict) -> dict:
    # Безопасное объединение реестров ссылок
    return {**left, **right}


def merge_classification_results(left: dict, right: dict) -> dict:
    # Объединяет mapped_data от Source Analyzer-ов, склеивая факты по главам
    merged = {**left}
    for key, value in right.items():
        if key in merged:
            merged[key] = merged[key] + value
        else:
            merged[key] = value
    return merged


def merge_drafts_reducer(left: dict, right: dict) -> dict:
    # Собирает готовые главы от параллельных писателей
    return {**left, **right}

class SourceFinderState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)
    url_registry: Annotated[dict[str, str], merge_registries] = Field(default_factory=dict)

    final_doc_ids: list[str] = Field(default_factory=list)

class FinalSources(BaseModel):
    doc_ids: list[str] = Field(
        description="Exact list of EVERY relevant doc_id. Ranges or abbreviations are strictly not allowed."
    )

class SourceAnalyzerState(BaseModel):
    doc_id: str
    url: str

    research_paper_plan: list[str] # should be enumerated

class SourceAnalyzerResponse(BaseModel):
    main_text: str = Field(description="Key takeaway (main context and important facts).")
    quotes: str = Field(description="Some precise quotes that might be helpful for argumentation")

    plan_indexes: list[int] = Field(description="List of 0-based integer indexes of chapters from the provided macro-plan that this article is directly relevant to.")


class PlanSection(BaseModel):
    name: str = Field(description="Name of the section")
    description: str = Field(description="Detailed description of the section")
    page_count: float = Field(description="Page count of the section")


class PlannerResponse(BaseModel):
    reasoning: str = Field(
        description="Your reasoning about the plan. Here you can count different blocks' pages..."
    )
    plan: list[PlanSection] = Field(
        description="Field for the plan. List items should represent paragraphs."
    )


class ResearchPaperState(BaseModel):
    research_topic: str
    research_length: int = 3  # страниц А4
    research_id: str

    # planner inputs
    academic_level: str = "college"
    additional_instructions: str = ""

    # planner output
    macro_plan: list[PlanSection] = Field(default_factory=list)

    # source finder output
    source_registry: dict[str, str] = Field(default_factory=dict)

    # source analyzer output (fan-in через редюсер)
    mapped_data: Annotated[dict[int, list], merge_classification_results] = Field(default_factory=dict)

    # scriptor output
    chapter_skeletons: dict[int, dict] = Field(default_factory=dict)

    # parallel writers output
    chapter_drafts: Annotated[dict[int, str], merge_drafts_reducer] = Field(default_factory=dict)

    # final output
    main_paper_text: Optional[str] = None


class ChapterSubgraphState(BaseModel):
    chapter_index: int
    chapter_name: str
    core_thesis: str
    bridge_to_next: str
    facts: list
    is_bibliography: bool = False
    is_intro_or_conclusion: bool = False
    
    page_count: float = 1.0
    academic_level: str = "college"
    additional_instructions: str = ""
    
    draft_text: Optional[str] = None
    reviewer_feedback: Optional[str] = None
    reviewer_result: Optional[Literal["needs_rewrite", "approved"]] = None
    iteration: int = 1
    chapter_drafts: Optional[dict] = None
