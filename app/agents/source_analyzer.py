from langchain_core.runnables import RunnableConfig
from langchain_core.messages import HumanMessage, SystemMessage
import asyncio

from app.config import cheap_llm

import logging
from app.schema import SourceAnalyzerState, SourceAnalyzerResponse
import httpx


async def extract_text_from_webpage(url: str, max_attempts=10):
    target_url = f"https://r.jina.ai/{url}"

    headers = {
        "Accept": "text/markdown",
    }

    for cur_attempt in range(1, max_attempts + 1):
        logging.info(f"Чтение информации со ссылки: {url} (Попытка {cur_attempt}/{max_attempts})")
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(target_url, headers=headers)
                response.raise_for_status()
                response.encoding = 'utf-8'
                return response.text[:40000]

        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logging.error(f"Ошибка при загрузке страницы {url}: {error_msg}")
            
            if isinstance(e, httpx.HTTPStatusError) and e.response.status_code in (401, 403, 404, 410, 451):
                logging.error(f"Фатальный HTTP статус {e.response.status_code}. Прерываем попытки загрузки.")
                return "ERROR_FAILED_TO_LOAD"
                
            if cur_attempt < max_attempts:
                logging.info(f"Сплю {cur_attempt * 2} секунд и повторяю загрузку...")
                await asyncio.sleep(cur_attempt * 2)
            else:
                logging.critical(f"Не удалось загрузить страницу {url} после {max_attempts} попыток.")
                # Возврат специального маркера ошибки вместо падения графа
                return "ERROR_FAILED_TO_LOAD"


async def analyze_webpage(url: str, enumerated_plan: str, max_attempts=10):
    # Извлечение и сжатие полного текста веб-страницы с помощью cheap_llm
    raw_text = await extract_text_from_webpage(url, max_attempts=max_attempts)
    if raw_text == "ERROR_FAILED_TO_LOAD": return raw_text

    # Формирование системного промпта с пронумерованным планом
    system_prompt = (
        "You are an expert academic research assistant. Your task is to analyze the provided web page content "
        "and map it to the specific chapters of a research paper plan.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. Extract a dense, high-quality summary in 'main_text' containing key arguments, data, and context. "
        "Do not lose the scientific meaning.\n"
        "2. Collect highly precise, unmodified 'quotes' that can serve as direct evidence or argumentation.\n"
        "3. Carefully evaluate the relevance of this text to each chapter in the provided macro-plan. "
        "Assign the chapter's integer index to 'plan_indexes' ONLY if the text contains explicit facts, "
        "data, or deep context for that chapter.\n"
        "4. ANTI-DUPLICATION RULE: Map each distinct fact or quote to EXACTLY ONE most relevant chapter index. "
        "DO NOT assign the same article to multiple chapters unless it contains completely different facts for each. "
        "If a document covers multiple chapters, the 'main_text' must explicitly state which facts belong to which chapter.\n"
        "5. IMPORTANT: Index 0 is the Introduction (Введение). Do NOT assign facts to the Introduction or Conclusion. "
        "These chapters will be synthesized from the body chapters later.\n\n"
        f"RESEARCH PAPER PLAN:\n{enumerated_plan}"
    )

    # Настройка структурированного вывода (SourceAnalyzerResponse)
    structured_cheap_llm = cheap_llm.with_structured_output(SourceAnalyzerResponse, method="function_calling")

    for cur_attempt in range(1, max_attempts + 1):
        try:
            # Вызов модели для сжатия текста и классификации фактов
            analyzer_result: SourceAnalyzerResponse = await structured_cheap_llm.ainvoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Текст для обработки:\n{raw_text}")
            ])

            logging.info(f"Текст успешно обработан и сжат моделью для индексов: {analyzer_result.plan_indexes}")
            return analyzer_result

        except Exception as e:
            logging.error(f"Ошибка при вызове cheap_llm для сжатия: {e}")
            if cur_attempt < max_attempts:
                logging.error(
                    f"Попытка {cur_attempt}/{max_attempts}. Сплю {cur_attempt * 2} секунд и повторяю запрос к LLM.")
                await asyncio.sleep(cur_attempt * 2)
            else:
                logging.critical(f"Критическая ошибка LLM при попытке #{cur_attempt}: {e}")
                return "ERROR_LLM_FAILED"


async def run_source_analyzer_node(state: SourceAnalyzerState, config: RunnableConfig = None) -> dict:
    if isinstance(state, dict):
        state = SourceAnalyzerState(**state)

    progress_queue = config.get("configurable", {}).get("progress_queue") if config else None
    if progress_queue:
        await progress_queue.put({"type": "custom", "message": f"🧠 Читаю и анализирую: {state.url[:40]}... ⏳"})

    enumerated_plan = "\n".join(f"[{idx}] {chapter}" for idx, chapter in enumerate(state.research_paper_plan))

    result = await analyze_webpage(state.url, enumerated_plan)

    # Возврат пустого словаря при фатальной ошибке источника (защита графа от зависания)
    if result in ["ERROR_FAILED_TO_LOAD", "ERROR_LLM_FAILED"]:
        logging.warning(f"Источник {state.doc_id} полностью провален. Пропускаем.")
        return {"mapped_data": {}}

    # Группировка результатов по индексу главы для reducer-а mapped_data
    classification_map = {}
    for idx in result.plan_indexes:
        if 0 <= idx < len(state.research_paper_plan):
            classification_map[idx] = [{
                "doc_id": state.doc_id,
                "url": state.url,
                "summary": result.main_text,
                "quotes": result.quotes
            }]

    return {"mapped_data": classification_map}
