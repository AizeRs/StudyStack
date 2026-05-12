from typing import Annotated, Optional, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field


class WriterState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

class ResearcherState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)

class ReviewerState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


class ResearchPaperState(BaseModel):
    research_topic: str

    research_id: int

    # from researcher
    raw_facts: Optional[str] = None

    #from writer
    main_paper_text: Optional[str] = None

    #from reviewer
    feedback: Optional[str] = None
    review_result_first: Optional[Literal["needs_facts", "needs_rewrite", "approved"]] = None

    score: Optional[float] = None


class FirstReviewerResponse(BaseModel):
    feedback: str = Field(
        description="Detailed, actionable feedback. Focus ONLY on macro-level issues: hallucinations, missing crucial data from raw_facts, or broken logical structure. Use bullet points for specific corrections."
    )
    scores: Annotated[list[Annotated[float, Field(ge=0, le=5)]], Field(min_length=5, max_length=5)] = Field(
        description="Exactly five scores (0.0 to 5.0) in this strict order: "
                    "1. Grounding (No hallucinations, aligns with raw_facts), "
                    "2. Factual Density (Proper use of data/quotes), "
                    "3. Completeness (Fully covers the research task), "
                    "4. Structure (Logical flow between paragraphs), "
                    "5. Objectivity (Maintains academic, neutral tone)."
    )
    review_result: Literal["needs_facts", "needs_rewrite", "approved"] = Field(
        description="Routing decision: "
                    "- 'needs_facts': The raw research data is fundamentally missing critical information required to answer the prompt. "
                    "- 'needs_rewrite': The data is sufficient, but the text fails the metrics (any score < 4.0) or contains hallucinations. "
                    "- 'approved': The text is solidly grounded, well-structured, and all scores are >= 4.0."
    )


