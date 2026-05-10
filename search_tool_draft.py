import requests
from pydantic import BaseModel, Field

from config import settings
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
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
    url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "text/event-stream",
    }

    response = requests.get(url, headers=headers)

    response.encoding = 'utf-8'

    logging.info(f"Результат чтения (первые 500 символов): \n {response.text[:500]}")
    return response.text[:15000]


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s"
    )

    llm = ChatOpenAI(
        model=settings.llm.model_name,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}}
    )

    researcher_llm = llm.bind_tools([search_web, read_webpage])
    messages = [HumanMessage(content="Найди информацию для доклада на тему: Исследование развития искусственного интелекта в период с 2024 по 2026 год. Рост рынка ИИ, числа пользователей, предоставляемого функционала, количества компаний связанных с ИИ. Анализ этих показателей.")]

    max_retries = 5
    cur_retry = 0

    while True:

        if cur_retry == max_retries:
            logging.warning("Лимит исчерпан. Форсируем ответ.")

            messages[
                -1].content += "\n\n[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Лимит поиска исчерпан. Сформируй итоговый ответ на основе этих данных."

            final_fallback_response = llm.invoke(messages)

            print(final_fallback_response.content)
            break

        response = researcher_llm.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            logging.info("Агент завершил работу и сформировал ответ.")
            print(response.content)
            break

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            logging.info(f"🛠️ Агент вызывает инструмент: {tool_name} с аргументами {tool_args}")

            if tool_name == "search_web":
                result = search_web.invoke(tool_args)
            elif tool_name == "read_webpage":
                result = read_webpage.invoke(tool_args)
            else:
                result = f"Ошибка: неизвестный инструмент {tool_name}"

            messages.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
        cur_retry += 1

if __name__ == "__main__":
    main()
