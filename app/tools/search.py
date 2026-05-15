import requests
from pydantic import BaseModel, Field

from app.config import cheap_llm
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from ddgs import DDGS
import logging


@tool
def search_web(query: str, max_results: int = 10) -> str:
    """
    Useful for searching for information on the internet.
    It accepts a search query and returns titles, snippets, and links (URLs) to websites.
    """
    logging.info(f"Поиск информации по теме {query}")
    try:
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append(f"Title: {r['title']}\nSnippet: {r['body']}\nURL: {r['href']}\n---")

        logging.info("Результат поиска:")
        if not results:
            logging.info("Ничего не найдено.")
            return "Ничего не найдено."
        logging.info(f"Найдено {len(results)} ссылок. Первая из них: {results[0]}")
        return "\n".join(results)
    except Exception as e:
        return f"Ошибка при поиске: {str(e)}"


class ReadWebpageArgs(BaseModel):
    url: str = Field(description="Только чистый URL-адрес, начинающийся с http:// или https://. Никакого текста вокруг.")


@tool(args_schema=ReadWebpageArgs)
def read_webpage(url: str):
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
        response = requests.get(target_url, headers=headers)
        response.raise_for_status()
        response.encoding = 'utf-8'
    except requests.RequestException as e:
        logging.error(f"Ошибка HTTP при запросе {url}: {e}")
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
        compressed_msg = cheap_llm.invoke([
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
