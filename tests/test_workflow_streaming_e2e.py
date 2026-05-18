import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from langchain_core.messages import AIMessage

from app.schema import FirstReviewerResponse, NoFactsFirstReviewerResponse
from app.workflow import stream_research_graph


# Полный E2E тест stream_research_graph
@pytest.mark.asyncio
@patch('app.agents.reviewer.llm')
@patch('app.agents.writer.llm')
@patch('app.tools.search.read_webpage.coroutine', new_callable=AsyncMock)
@patch('app.tools.search.search_web.coroutine', new_callable=AsyncMock)
@patch('app.agents.researcher.llm')
async def test_stream_research_graph_e2e(
        mock_researcher_llm: MagicMock,
        mock_search_web_coro: AsyncMock,
        mock_read_webpage_coro: AsyncMock,
        mock_writer_llm: MagicMock,
        mock_reviewer_llm: MagicMock
):
    # 1. Настраиваем ресерчера (делает два вызова тулов, затем выдает факты)
    mock_llm_with_tools = MagicMock()
    mock_res_ainvoke = AsyncMock()
    mock_res_ainvoke.side_effect = [
        AIMessage(content="", tool_calls=[{"name": "search_web", "args": {"query": "AI"}, "id": "call_1"}]),
        AIMessage(content="", tool_calls=[{"name": "read_webpage", "args": {"url": "http://ai.com"}, "id": "call_2"}]),
        AIMessage(content="Final raw facts about AI.")
    ]
    mock_llm_with_tools.ainvoke = mock_res_ainvoke
    mock_researcher_llm.bind_tools.return_value = mock_llm_with_tools

    mock_search_web_coro.return_value = "Search results"
    mock_read_webpage_coro.return_value = "Page content"

    # 2. Настраиваем писателя (сначала пишет плохо, после фидбека - хорошо)
    mock_writer_ainvoke = AsyncMock()
    mock_writer_ainvoke.side_effect = [
        AIMessage(content="Bad Draft"),
        AIMessage(content="Perfect Draft")
    ]
    mock_writer_llm.ainvoke = mock_writer_ainvoke

    # 3. Настраиваем критика (сначала бракует, затем одобряет)
    mock_structured_llm = MagicMock()
    mock_rev_ainvoke = AsyncMock()
    mock_rev_ainvoke.side_effect = [
        FirstReviewerResponse(
            feedback="Needs more logic", scores=[3.0, 3.0, 3.0, 3.0, 3.0], review_result="needs_rewrite"
        ),
        NoFactsFirstReviewerResponse(
            feedback="Excellent", scores=[5.0, 5.0, 5.0, 5.0, 5.0], review_result="approved"
        )
    ]
    mock_structured_llm.ainvoke = mock_rev_ainvoke
    mock_reviewer_llm.with_structured_output.return_value = mock_structured_llm

    # ЗАПУСК СТРИМИНГА РЕАЛЬНОГО ГРАФА
    events = []
    async for node_name, state_update in stream_research_graph("AI History", research_id="stream_100"):
        events.append((node_name, state_update))

    # ПРОВЕРКИ
    # Ожидаемый поток:
    # 1. researcher отработал
    # 2. writer написал первый черновик
    # 3. macro_reviewer забраковал (needs_rewrite)
    # 4. writer написал второй черновик
    # 5. macro_reviewer одобрил (approved)
    assert len(events) == 5

    # 1-й этап: Сбор фактов
    assert events[0][0] == "researcher"
    assert "raw_facts" in events[0][1]
    assert "Final raw facts" in events[0][1]["raw_facts"]

    # 2-й этап: Первый черновик
    assert events[1][0] == "writer"
    assert events[1][1]["main_paper_text"] == "Bad Draft"

    # 3-й этап: Первое ревью
    assert events[2][0] == "macro_reviewer"
    assert events[2][1]["macro_reviewer_result"] == "needs_rewrite"
    assert events[2][1]["macro_reviewer_iteration"] == 2

    # 4-й этап: Исправленный черновик
    assert events[3][0] == "writer"
    assert events[3][1]["main_paper_text"] == "Perfect Draft"

    # 5-й этап: Финальное ревью
    assert events[4][0] == "macro_reviewer"
    assert events[4][1]["macro_reviewer_result"] == "approved"

    # Убеждаемся, что тулы действительно были вызваны на слое ресерчера
    assert mock_search_web_coro.call_count == 1
    assert mock_read_webpage_coro.call_count == 1
