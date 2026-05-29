import logging
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field

from app.config import llm
from app.schema import ResearchPaperState

class ScriptorResponse(BaseModel):
    core_thesis: str = Field(description="The core skeleton/thesis for this chapter. 1-2 paragraphs summarizing the facts seamlessly.")
    bridge_to_next: str = Field(description="A transitional sentence that bridges logically to the next chapter.")

async def run_scriptor_node(state: ResearchPaperState, config: RunnableConfig = None) -> dict:
    logging.info("СТАРТ УЗЛА: Итеративный Сценарист (Scriptor)")
    
    progress_queue = config.get("configurable", {}).get("progress_queue") if config else None
    
    plan = state.macro_plan
    num_chapters = len(plan)
    if num_chapters == 0:
        return {"chapter_skeletons": {}}

    skeletons = {}
    structured_llm = llm.with_structured_output(ScriptorResponse, method="function_calling")
    
    # Определение индексов специальных глав
    biblio_idx = num_chapters - 1 if num_chapters > 0 else -1
    conclusion_idx = num_chapters - 2 if num_chapters > 1 else -1
            
    # Генерация скелетов для основных глав
    previous_bridge = ""
    for i in range(1, conclusion_idx):
        logging.info(f"[Scriptor] Генерация скелета для главы {i} ({plan[i].name})")
        if progress_queue:
            await progress_queue.put({
                "type": "chapter_status",
                "phase": "skeleton",
                "chapter_index": i,
                "chapter_name": plan[i].name,
                "status": "⏳ В работе"
            })
            
        chapter_facts = state.mapped_data.get(i, [])
        next_chapter_name = plan[i+1].name if i + 1 < num_chapters else "Conclusion"
        
        prompt = f"""Role: You are the Master Scriptor. You are writing the structural skeleton for Chapter {i}: "{plan[i].name}".
        
Here is the transitional bridge from the previous chapter:
{previous_bridge if previous_bridge else "This is the first body chapter. Start directly."}

Here are the facts assigned to this chapter:
{chapter_facts}

Task: Write the `core_thesis` (1-2 paragraphs synthesizing these facts logically) and a `bridge_to_next` (1 sentence transitioning smoothly to the next chapter: "{next_chapter_name}").
If the facts list is empty, rely on your general academic knowledge and the structural logic of the paper to formulate the thesis.
Do not hallucinate. Use only the provided facts if they are present.
"""
        response = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        skeletons[i] = {
            "core_thesis": response.core_thesis,
            "bridge_to_next": response.bridge_to_next
        }
        previous_bridge = response.bridge_to_next

    # Генерация скелета Введения (на основе основных глав)
    if num_chapters > 0:
        logging.info("[Scriptor] Генерация Введения (глава 0)")
        if progress_queue:
            await progress_queue.put({"type": "custom", "message": f"✍️ Сценарист продумывает Введение (главу 0)... ⏳"})
        body_summary = "\n".join([f"Chapter {i} ({plan[i].name}): {skeletons[i]['core_thesis']}" for i in range(1, conclusion_idx)]) if conclusion_idx > 1 else ""
        
        next_chapter_name = plan[1].name if num_chapters > 1 else "the end of the paper"
        intro_prompt = f"""Role: You are the Master Scriptor writing the Introduction (Chapter 0: "{plan[0].name}").

Here is the entire skeleton of the main body of the paper (FOR CONTEXT ONLY, DO NOT REPEAT THIS):
{body_summary}

Here are specific facts assigned to the introduction (if any):
{state.mapped_data.get(0, [])}

Task: Write the `core_thesis` for the Introduction.
CRITICAL: Focus ONLY on outlining the problem statement, relevance of the topic, and briefly outlining the structure of the paper. DO NOT reveal the detailed statistics, arguments, or conclusions from the main body. 
Also write a `bridge_to_next` transitioning to Chapter 1 ("{next_chapter_name}").
"""
        response_intro = await structured_llm.ainvoke([HumanMessage(content=intro_prompt)])
        skeletons[0] = {
            "core_thesis": response_intro.core_thesis,
            "bridge_to_next": response_intro.bridge_to_next
        }

    # Генерация скелета Заключения
    if conclusion_idx > 0:
        last_idx = conclusion_idx
        logging.info(f"[Scriptor] Генерация Заключения (глава {last_idx})")
        if progress_queue:
            await progress_queue.put({"type": "custom", "message": f"✍️ Сценарист продумывает Заключение (главу {last_idx})... ⏳"})
        conclusion_prompt = f"""Role: You are the Master Scriptor writing the Conclusion (Chapter {last_idx}: "{plan[last_idx].name}").

Here is the entire skeleton of the main body of the paper (FOR SYNTHESIS ONLY, DO NOT REPEAT RAW FACTS):
{body_summary}

Here are specific facts assigned to the conclusion (if any):
{state.mapped_data.get(last_idx, [])}

Here is the bridge from the last body chapter:
{previous_bridge}

Task: Write the `core_thesis` for the Conclusion.
CRITICAL: Synthesize the overarching meaning of the findings. Provide final thoughts, implications, and future outlook. DO NOT repeat the exact statistics or detailed arguments from the body chapters.
Set `bridge_to_next` to an empty string.
"""
        response_concl = await structured_llm.ainvoke([HumanMessage(content=conclusion_prompt)])
        skeletons[last_idx] = {
            "core_thesis": response_concl.core_thesis,
            "bridge_to_next": response_concl.bridge_to_next
        }

    # Формирование псевдо-скелета для библиографии
    if biblio_idx != -1:
        logging.info(f"[Scriptor] Генерация Библиографии (глава {biblio_idx})")
        if progress_queue:
            await progress_queue.put({"type": "custom", "message": f"✍️ Сценарист формирует список литературы... ⏳"})
        skeletons[biblio_idx] = {
            "core_thesis": "Format the provided sources into a proper academic bibliography/reference list.",
            "bridge_to_next": ""
        }

    logging.info("ЗАВЕРШЕНИЕ УЗЛА: Итеративный Сценарист")
    return {"chapter_skeletons": skeletons}
