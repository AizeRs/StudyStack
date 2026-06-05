import asyncio
from uuid import uuid4

from langchain_core.tools import tool
from ddgs import DDGS
import logging


def _fetch_ddgs(query: str, max_results: int):
    """
    Синхронная функция для запроса к DuckDuckGo.
    Вынесена отдельно для безопасного запуска в пуле потоков (asyncio.to_thread),
    так как библиотека DDGS выполняет блокирующие сетевые вызовы.
    """
    results = []
    with DDGS(timeout=30) as ddgs:
        # Исключаем brave и mojeek, чтобы избежать массовых 429
        safe_backends = "duckduckgo,google,yahoo,wikipedia,startpage,yandex"
        for r in ddgs.text(query, max_results=max_results, backend=safe_backends):
            results.append(r)
    return results


@tool(response_format="content_and_artifact")
async def search_web(query: str, max_results: int = 10) -> tuple[str, dict]:
    """
    Инструмент поиска информации в интернете (DuckDuckGo).
    Реализует механизм экспоненциальной задержки (Exponential Backoff) для защиты от Rate Limit (429).
    
    Возвращает кортеж:
    - Отформатированный текст для передачи в LLM.
    - Артефакт: словарь (registry) с привязкой сгенерированных doc_id к реальным URL.
    """
    logging.info(f"Поиск информации по теме {query}")
    max_retries = 3
    base_delay = 3
    
    for attempt in range(max_retries):
        try:
            results = await asyncio.to_thread(_fetch_ddgs, query, max_results)
    
            if not results:
                return "Ничего не найдено.", {}
    
            results_text = []
            url_registry = {}
    
            for r in results:
                doc_id = f"doc_{uuid4().hex[:8]}"
                url_registry[doc_id] = r['href']
                results_text.append(f"ID: {doc_id}\nTitle: {r['title']}\nSnippet: {r['body']}\n---")
    
            text_for_llm = "\n".join(results_text)
    
            # Кортеж: (текст, артефакт)
            return text_for_llm, url_registry
    
        except Exception as e:
            error_str = str(e)
            if attempt < max_retries - 1 and ("429" in error_str or "Timeout" in error_str or "ReadTimeout" in error_str):
                sleep_time = base_delay * (2 ** attempt)
                logging.warning(f"Ошибка поиска {error_str}. Повтор {attempt + 1}/{max_retries} через {sleep_time}с...")
                await asyncio.sleep(sleep_time)
                continue
            
            logging.error(f"Финальная ошибка при поиске: {error_str}")
            return f"Ошибка при поиске: {error_str}", {}



