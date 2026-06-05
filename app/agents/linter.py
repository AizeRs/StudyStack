import logging
import re
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from app.config import llm
from app.schema import ChapterSubgraphState

SYSTEM_PROMPT_LINTER = """Role: Markdown Linter.
Your objective is to fix the Markdown formatting of the provided text.
Rules:
1. Ensure proper use of headings (H2, H3) where appropriate.
2. Ensure bolding and italics are used appropriately to emphasize key points.
3. Ensure lists are formatted correctly.
4. Do NOT change the meaning, facts, or flow of the text.
5. Return ONLY the formatted text. Do not include markdown code block wrappers (like ```markdown).
"""

async def run_linter_node(state: ChapterSubgraphState, config: RunnableConfig = None) -> dict:
    logging.info(f"[Linter] Форматирование главы {state.chapter_index}")
    
    progress_queue = config.get("configurable", {}).get("progress_queue") if config else None
    if progress_queue:
        await progress_queue.put({
            "type": "chapter_status",
            "phase": "drafting",
            "chapter_index": state.chapter_index,
            "chapter_name": state.chapter_name,
            "status": "✨ Линтер (форматирование)"
        })
    
    context_prompt = (
        f"Context: This is Chapter {state.chapter_index}: '{state.chapter_name}'.\n"
        f"CRITICAL RULES FOR HEADINGS:\n"
        f"1. The text MUST start with a level 2 heading (##) containing the chapter name.\n"
        f"   Example: `## {state.chapter_index}. {state.chapter_name}` (If index is 0 or this is Conclusion/Bibliography, you can omit the number and just use `## {state.chapter_name}`).\n"
        f"2. Any internal sub-sections MUST use level 3 headings (###) or lower. NEVER use # or ## for sub-sections."
    )
    
    messages = [
        SystemMessage(content=SYSTEM_PROMPT_LINTER + "\n\n" + context_prompt),
        HumanMessage(content=f"Text to format:\n\n{state.draft_text}")
    ]
    
    response = await llm.ainvoke(messages)
    
    clean_text = response.content.strip()
    if clean_text.startswith("```"):
        clean_text = re.sub(r"^```[a-zA-Z]*\n?", "", clean_text)
    if clean_text.endswith("```"):
        clean_text = re.sub(r"\n?```$", "", clean_text)
    clean_text = clean_text.strip()
    
    if progress_queue:
        await progress_queue.put({
            "type": "chapter_status",
            "phase": "drafting",
            "chapter_index": state.chapter_index,
            "chapter_name": state.chapter_name,
            "status": "✅ Готово"
        })
    
    return {
        "draft_text": clean_text,
        "chapter_drafts": {state.chapter_index: clean_text}
    }
