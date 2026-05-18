import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.tools.search import search_web, read_webpage
import httpx
from langchain_core.messages import AIMessage


# -------- ТЕСТЫ ДЛЯ search_web --------

@pytest.mark.asyncio
@patch('app.tools.search._fetch_ddgs')
async def test_search_web_success(mock_fetch_ddgs: MagicMock):
    # Успешный ответ от DDGS
    mock_fetch_ddgs.return_value = [
        "Title: Pizza\nSnippet: Order tasty pizza.\nURL: https://pizza.com/1\n---",
        "Title: We also have sushi\nSnippet: Order tasty sushi.\nURL: https://pizza.com/2\n---"
    ]

    result = await search_web.ainvoke({"query": "Order pizza or sushi", "max_results": 2})

    assert "Title: Pizza" in result
    assert "Title: We also have sushi" in result
    assert result.count("---") == 2
    mock_fetch_ddgs.assert_called_once_with("Order pizza or sushi", 2)


@pytest.mark.asyncio
@patch('app.tools.search._fetch_ddgs')
async def test_search_web_empty_result(mock_fetch_ddgs: MagicMock):
    # DDGS ничего не нашёл
    mock_fetch_ddgs.return_value = []

    result = await search_web.ainvoke({"query": "free pizza"})

    assert result == "Ничего не найдено."
    mock_fetch_ddgs.assert_called_once()


@pytest.mark.asyncio
@patch('app.tools.search._fetch_ddgs')
async def test_search_web_exception_handling(mock_fetch_ddgs: MagicMock):
    # Падение внутри DDGS
    mock_fetch_ddgs.side_effect = Exception("DDGS rate limit exceeded")

    result = await search_web.ainvoke({"query": "a billion pizzas"})

    assert "Ошибка при поиске: DDGS rate limit exceeded" in result
    mock_fetch_ddgs.assert_called_once()

# -------- ТЕСТЫ ДЛЯ read_webpage --------

@pytest.mark.asyncio
@patch('app.tools.search.cheap_llm')
@patch('app.tools.search.httpx.AsyncClient')
async def test_read_webpage_success(mock_httpx_client, mock_cheap_llm):
    # Успешный ответ и от api джины, и от cheap_llm

    # 1. Настраиваем фейковый ответ от httpx
    mock_response = MagicMock()
    mock_response.text = "Это очень длинный текст " * 1000
    mock_response.raise_for_status.return_value = None

    # 2. Настраиваем асинхронный контекстный менеджер (async with)
    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_response
    mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance

    # 3. Настраиваем успешный ответ от cheap_llm
    mock_ainvoke = AsyncMock()
    mock_ainvoke.return_value = AIMessage(content="Сжатый текст с фактами и цитатами.")
    mock_cheap_llm.ainvoke = mock_ainvoke

    result = await read_webpage.ainvoke({"url": "example.com"})

    assert result == "Сжатый текст с фактами и цитатами."
    mock_client_instance.get.assert_called_once_with(
        "https://r.jina.ai/example.com",
        headers={"Accept": "text/event-stream"}
    )
    mock_cheap_llm.ainvoke.assert_called_once()


@pytest.mark.asyncio
@patch('app.tools.search.httpx.AsyncClient')
async def test_read_webpage_http_status_error(mock_httpx_client):
    # Джина отвечает с ошибкой
    mock_request = MagicMock(url="https://r.jina.ai/example.com")
    mock_response = MagicMock(status_code=404)

    error = httpx.HTTPStatusError("404 Not Found", request=mock_request, response=mock_response)

    mock_client_instance = AsyncMock()
    mock_client_instance.get.side_effect = error
    mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance

    result = await read_webpage.ainvoke({"url": "example.com"})

    assert "Не удалось загрузить страницу. Ошибка:" in result
    assert "404 Not Found" in result


@pytest.mark.asyncio
@patch('app.tools.search.httpx.AsyncClient')
async def test_read_webpage_request_error(mock_httpx_client):
    # Отвал сети (или падение сервера джины)
    mock_request = MagicMock(url="https://r.jina.ai/example.com")
    error = httpx.RequestError("Connection timeout", request=mock_request)

    mock_client_instance = AsyncMock()
    mock_client_instance.get.side_effect = error
    mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance

    result = await read_webpage.ainvoke({"url": "example.com"})

    assert "Не удалось загрузить страницу. Ошибка:" in result
    assert "Connection timeout" in result


@pytest.mark.asyncio
@patch('app.tools.search.cheap_llm')
@patch('app.tools.search.httpx.AsyncClient')
async def test_read_webpage_llm_rate_limit(mock_httpx_client, mock_cheap_llm):
    # Успешный ответ от джины и падение cheap_llm

    mock_response = MagicMock()
    mock_response.text = "Много текста"
    mock_response.raise_for_status.return_value = None

    mock_client_instance = AsyncMock()
    mock_client_instance.get.return_value = mock_response
    mock_httpx_client.return_value.__aenter__.return_value = mock_client_instance

    mock_ainvoke = AsyncMock()
    mock_ainvoke.side_effect = Exception("OpenAI Rate Limit Exceeded")
    mock_cheap_llm.ainvoke = mock_ainvoke

    result = await read_webpage.ainvoke({"url": "example.com"})

    # Должен сработать фолбэк-ответ
    assert result == "Доступ к read_webpage временно ограничен (Rate Limit Exceeded). Используйте для отчёта уже собранные сведения."


