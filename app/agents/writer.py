from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
import logging
from app.config import llm
from app.schema import ChapterSubgraphState

SYSTEM_PROMPT_WRITER_TEMPLATE = """Role: Expert Academic Chapter Writer.
Mission: Expand the provided `core_thesis` into a detailed, focused chapter using the provided facts.
Rules:
1. Target Audience/Level: Write in a style appropriate for a {academic_level}.
2. Target Length: You must write approximately {word_count} words to satisfy the assigned page count requirement of {page_count} pages for this specific chapter.
3. Special Instructions: Strictly follow these user instructions: {additional_instructions}
4. Fidelity & Scope: Use ONLY the provided facts. You are writing ONE specific chapter of a larger paper, NOT the entire paper. DO NOT summarize or anticipate other chapters. 
5. Introduction/Conclusion Rule: If you are writing the Introduction, focus ONLY on the problem statement, relevance, and outlining the paper structure; DO NOT reveal detailed stats or findings. If you are writing the Conclusion, synthesize the main takeaways briefly without repeating raw numbers.
6. Structure: Follow the `core_thesis` as your main narrative skeleton. Do not deviate into unrelated topics.
7. Transition: End the chapter smoothly incorporating the `bridge_to_next` concept so the next chapter flows naturally.
8. Output: Return ONLY the chapter text, without introductory filler or markdown code blocks.
"""

SYSTEM_PROMPT_BIBLIOGRAPHY = """Role: Expert Academic Bibliography Formatter.
Mission: Format the provided facts (which are URLs or raw sources) into a proper academic reference list.
Rules:
1. Output ONLY a clean, bulleted or numbered list of the sources.
2. DO NOT write any introductory or concluding text, meta-essays, or explanations. Just the list.
3. If the sources are URLs, format them nicely.
4. Output: Return ONLY the formatted list, without markdown code blocks.
"""

async def run_writer_node(state: ChapterSubgraphState, config: RunnableConfig = None) -> dict:
    logging.info(f"[Writer] Генерация текста для главы {state.chapter_index}: {state.chapter_name}")
    
    progress_queue = config.get("configurable", {}).get("progress_queue") if config else None
    if progress_queue:
        if state.reviewer_feedback and state.reviewer_result == "needs_rewrite":
            await progress_queue.put({"type": "custom", "message": f"♻️ Переписываю главу {state.chapter_index} ({state.chapter_name}) по замечаниям редактора... ⏳"})
        else:
            await progress_queue.put({"type": "custom", "message": f"🚀 Пишу текст: глава {state.chapter_index} ({state.chapter_name})... ⏳"})
    
    if not state.facts and not state.is_bibliography and not getattr(state, "is_intro_or_conclusion", False):
        if progress_queue:
            await progress_queue.put({
                "type": "chapter_status",
                "phase": "drafting",
                "chapter_index": state.chapter_index,
                "chapter_name": state.chapter_name,
                "status": "⚠️ Пропуск (нет данных)"
            })
        return {"draft_text": "<!-- НЕ НАЙДЕНО ФАКТОВ ДЛЯ ДАННОЙ ГЛАВЫ -->\n\n*В связи с недостатком релевантных источников, детализированная генерация данной главы была пропущена системой.*"}
        
    if state.is_bibliography:
        system_content = SYSTEM_PROMPT_BIBLIOGRAPHY
    else:
        words = int(state.page_count * 400) if state.page_count > 0 else 400
        instructions = state.additional_instructions if state.additional_instructions else "None"
        system_content = SYSTEM_PROMPT_WRITER_TEMPLATE.format(
            academic_level=state.academic_level,
            word_count=words,
            page_count=state.page_count,
            additional_instructions=instructions
        )

    if state.reviewer_feedback and state.reviewer_result == "needs_rewrite":
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"Previous Draft:\n{state.draft_text}\n\nReviewer Feedback:\n{state.reviewer_feedback}\n\nRewrite the chapter to fix these issues. Maintain the core thesis and bridge.")
        ]
    else:
        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=f"""Chapter: {state.chapter_name}
Core Thesis: {state.core_thesis}
Facts: {state.facts}
Bridge to next chapter: {state.bridge_to_next}

Write the chapter now.""")
        ]
        
    response = await llm.ainvoke(messages)
    return {"draft_text": response.content}
