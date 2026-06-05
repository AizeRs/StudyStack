from pydantic import BaseModel, Field
from typing import Literal
import logging
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from app.config import llm
from app.schema import ChapterSubgraphState

class ChapterReviewResponse(BaseModel):
    feedback: str = Field(description="Detailed feedback if the text fails, focusing on missing facts or hallucinations.")
    review_result: Literal["needs_rewrite", "approved"] = Field(description="'approved' if facts are correct and thesis is maintained. 'needs_rewrite' otherwise.")

SYSTEM_PROMPT_REVIEWER = """Role: Academic Fact-Checker.
Your objective is to evaluate a drafted chapter against the provided facts, core thesis, and bridge.
Rules:
1. Ensure NO hallucinations. The draft must only use the provided facts.
2. Ensure the `core_thesis` is well-represented.
3. Ensure the `bridge_to_next` is naturally integrated at the end.
4. Output strict JSON with your evaluation.
"""

async def run_reviewer_node(state: ChapterSubgraphState, config: RunnableConfig = None) -> dict:
    logging.info(f"[Reviewer] Проверка главы {state.chapter_index}")
    
    progress_queue = config.get("configurable", {}).get("progress_queue") if config else None
    if progress_queue:
        await progress_queue.put({
            "type": "chapter_status",
            "phase": "drafting",
            "chapter_index": state.chapter_index,
            "chapter_name": state.chapter_name,
            "status": "🧐 Проверка редактором"
        })
        
    if not state.facts and not state.is_bibliography:
        logging.info(f"[Reviewer] Пропуск проверки для главы {state.chapter_index} (нет фактов)")
        return {
            "reviewer_feedback": "Skipped (No facts available).",
            "reviewer_result": "approved",
            "iteration": state.iteration + 1
        }
    
    structured_llm = llm.with_structured_output(ChapterReviewResponse, method="function_calling")
    
    prompt = f"""Chapter: {state.chapter_name}
Expected Core Thesis: {state.core_thesis}
Expected Bridge to Next: {state.bridge_to_next}
Raw Facts: {state.facts}

---DRAFT TEXT---
{state.draft_text}
---END DRAFT---

Evaluate the draft."""
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_REVIEWER),
        HumanMessage(content=prompt)
    ]
    
    response = await structured_llm.ainvoke(messages)
    
    logging.info(f"[Reviewer] Вердикт: {response.review_result}")
    
    return {
        "reviewer_feedback": response.feedback,
        "reviewer_result": response.review_result,
        "iteration": state.iteration + 1
    }