import asyncio
import uuid

from pydantic import BaseModel, Field
from app.config import cheap_llm
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS
import httpx
import logging


def _fetch_ddgs(query: str, max_results: int):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(r)
    return results


@tool(response_format="content_and_artifact")
async def search_web(query: str, max_results: int = 10) -> tuple[str, dict]:
    """
    Useful for searching for information on the internet.
    Returns doc_ids and snippets.
    """
    logging.info(f"Поиск информации по теме {query}")
    try:
        results = await asyncio.to_thread(_fetch_ddgs, query, max_results)

        if not results:
            return "Ничего не найдено.", {}

        results_text = []
        url_registry = {}

        for r in results:
            doc_id = f"doc_{uuid.uuid4().hex[:8]}"
            url_registry[doc_id] = r['href']
            results_text.append(f"ID: {doc_id}\nTitle: {r['title']}\nSnippet: {r['body']}\n---")

        text_for_llm = "\n".join(results_text)

        # Кортеж: (текст, артефакт)
        return text_for_llm, url_registry

    except Exception as e:
        return f"Ошибка при поиске: {str(e)}", {}


class ReadWebpageArgs(BaseModel):
    url: str = Field(description="Только чистый URL-адрес, начинающийся с http:// или https://. Никакого текста вокруг.")


@tool(args_schema=ReadWebpageArgs)
async def read_webpage(url: str):
    """
    Extracts the full text content of a web page.
    It accepts the URL of the page (which you previously found via search_web)
    and returns the clean text in Markdown format.
    The url field only accepts URLs, without formatting or other information.
    """
    logging.info(f"Чтение информации со ссылки: {url}")
    target_url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "text/event-stream",
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(target_url, headers=headers)
            response.raise_for_status()
            response.encoding = 'utf-8'

    except httpx.HTTPStatusError as e:
        # Сервер доступен, но вернул ошибку
        logging.error(f"Ошибка ответа: Сервер вернул {e.response.status_code} при запросе к {e.request.url}")
        return f"Не удалось загрузить страницу. Ошибка: {str(e)}"

    except httpx.RequestError as e:
        # Проблемы с сетью: сервер недоступен, таймаут, отвал DNS
        logging.error(f"Ошибка сети: Произошла ошибка при запросе к {e.request.url}. Детали: {e}")
        return f"Не удалось загрузить страницу. Ошибка: {str(e)}"

    except httpx.HTTPError as e:
        # Глобальный fallback для любых других непредвиденных ошибок httpx
        print(f"")
        logging.error(f"Критическая ошибка HTTPX: {e}")
        return f"Не удалось загрузить страницу. Ошибка: {str(e)}"

    raw_text = response.text[:40000]
    # logging.info(f"Результат чтения (первые 500 символов): \n {raw_text[:500]}")

    system_prompt = (
        "Ты — исследовательский ассистент. Твоя задача — сжать предоставленный текст веб-страницы, "
        "выделив только главную суть и факты, чтобы сэкономить токены.\n\n"
        "КРИТИЧЕСКОЕ УСЛОВИЕ: Ты обязан извлекать и сохранять важные оригинальные формулировки, "
        "статистику, термины и ключевые утверждения в виде точных цитат (в кавычках). "
        "Основная модель будет использовать твой ответ для написания академического текста, "
        "поэтому ей нужны прямые цитаты из источника.\n\n"
        "Структура ответа:\n"
        "1. Основная выжимка (контекст и факты).\n"
        "2. Точные цитаты, которые могут понадобиться для аргументации."
    )

    try:
        # Вызов дешевой модели для сжатия
        compressed_msg = await cheap_llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Текст для обработки:\n{raw_text}")
        ])

        compressed_text = compressed_msg.content
        logging.info(f"Текст сжат. Токенов сэкономлено: длина до {len(raw_text)}, после {len(compressed_text)}")
        return compressed_text

    except Exception as e:
        logging.error(f"Ошибка при вызове cheap_llm для сжатия: {e}")
        # Фолбэк: если LLM упала (например, RateLimit), возвращаем текст-заглушку
        return "Доступ к read_webpage временно ограничен (Rate Limit Exceeded). Используйте для отчёта уже собранные сведения."
