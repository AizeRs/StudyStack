from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphRecursionError

from schema import ResearchPaperState
import researcher_subgraph

# Патчим метод invoke напрямую в модуле, где тулы были определены
# Это единственный выход, так как добавление оригинальных функций тулов в файле ресерчера происходит
# сразу же во время импортирования функции run_researcher_subgraph_node в файл с тестами
# и патчи @patch('langgraph_researcher_draft.search_web') не успевают сработать
@patch('search_tool_draft.read_webpage.func')
@patch('search_tool_draft.search_web.func')
@patch('researcher_subgraph.llm')
def test_run_researcher_subgraph_node(mock_llm: MagicMock,
                                      mock_search_web_func: MagicMock,
                                      mock_read_webpage_func: MagicMock):
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.invoke.side_effect = [
        AIMessage(content="", tool_calls=[
            {
                "name": "search_web",
                "args": {"query": "order pizza"},
                "id": "1"
            }]),
        AIMessage(content="", tool_calls=[
            {
                "name": "search_web",
                "args": {"query": "order hot delicious pizza"},
                "id": "2"
            }]),
        AIMessage(content="", tool_calls=[
            {
                "name": "read_webpage",
                "args": {"url": "https://pizzahut.com/order-the-best-pizza"},
                "id": "3"
            }]),
        AIMessage(content="Facts for your research: the best pizza is made by pizza hut.")
    ]

    mock_llm.bind_tools.return_value = mock_llm_with_tools

    mock_search_web_func.return_value = "Some webpages"
    mock_read_webpage_func.return_value = "Some interesting text"

    state = ResearchPaperState(research_topic="where can i buy the best pizza?", research_id=1)
    config: RunnableConfig = {"configurable": {"researcher_recursion_limit": 10}}

    result = researcher_subgraph.run_researcher_subgraph_node(state, config)
    assert result == {"raw_facts": "Facts for your research: the best pizza is made by pizza hut."}
    mock_search_web_func.assert_called()
    mock_read_webpage_func.assert_called_once()


# Инкрементальный поиск (Добавление фактов)
@patch('search_tool_draft.read_webpage.func')
@patch('search_tool_draft.search_web.func')
@patch('researcher_subgraph.llm')
def test_incremental_search_branch(mock_llm, mock_search, mock_read):
    mock_llm_with_tools = MagicMock()
    mock_llm_with_tools.invoke.side_effect = [
        AIMessage(content="Специфичные новые факты по запросу критика.")
    ]
    mock_llm.bind_tools.return_value = mock_llm_with_tools

    state = ResearchPaperState(
        research_topic="pizza",
        research_id=1,
        raw_facts="Старые базовые факты.",
        macro_reviewer_result="needs_facts",
        macro_reviewer_feedback="Узнай точный год основания Pizza Hut.",
        facts_version=1
    )

    result = researcher_subgraph.run_researcher_subgraph_node(state, {})

    # Проверяем, что новые факты приклеились к старым
    assert "Старые базовые факты." in result["raw_facts"]
    assert "--- ADDITIONAL RESEARCH DATA (PARTIAL) ---" in result["raw_facts"]
    assert "Специфичные новые факты по запросу критика." in result["raw_facts"]

    assert result["extra_research_forbidden"] is True
    assert result["facts_version"] == 2
    assert result["macro_reviewer_iteration"] == 1


# Ошибка рекурсии (Graceful shutdown)
@patch('researcher_subgraph.app.invoke')
@patch('researcher_subgraph.app.get_state')
@patch('researcher_subgraph.llm')
def test_recursion_error_graceful_shutdown(mock_llm, mock_get_state, mock_app_invoke):
    mock_app_invoke.side_effect = GraphRecursionError("Recursion limit")

    # Симулируем стейт, где последним сообщением был ответ от тула
    mock_state = MagicMock()
    mock_state.values = {
        "messages": [
            HumanMessage(content="Start"),
            ToolMessage(content="Результаты поиска про пиццу", tool_call_id="1")
        ]
    }
    mock_get_state.return_value = mock_state

    # Фолбэк-ответ модели на основе обрезанных данных через mock_llm.invoke
    mock_llm.invoke.return_value = AIMessage(content="Финальная выжимка после обрыва")

    state = ResearchPaperState(research_topic="pizza", research_id=1)
    result = researcher_subgraph.run_researcher_subgraph_node(state, {})

    # Проверяем, что вернулся результат от LLM, а не упала ошибка
    assert result["raw_facts"] == "Финальная выжимка после обрыва"

    # Проверяем, что к ToolMessage реально приклеилось системное уведомление
    called_messages = mock_llm.invoke.call_args[0][0]
    assert "[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Лимит поиска исчерпан" in called_messages[-1].content


# Ошибка рекурсии (Обрезка зависшего AIMessage)
@patch('researcher_subgraph.app.invoke')
@patch('researcher_subgraph.app.get_state')
@patch('researcher_subgraph.llm')
def test_recursion_error_ai_message_pop_and_incremental(mock_llm, mock_get_state, mock_app_invoke):
    mock_app_invoke.side_effect = GraphRecursionError("Recursion limit")

    # Симулируем стейт, где граф упал сразу после того, как AI захотел вызвать тул
    mock_state = MagicMock()
    mock_state.values = {
        "messages": [
            HumanMessage(content="Start"),
            ToolMessage(content="Полезная инфа", tool_call_id="1"),
            AIMessage(content="", tool_calls=[{"name": "search_web", "args": {}, "id": "2"}])
        ]
    }
    mock_get_state.return_value = mock_state
    mock_llm.invoke.return_value = AIMessage(content="Экстренные новые факты")

    state = ResearchPaperState(
        research_topic="pizza",
        research_id=1,
        raw_facts="Старые факты",
        macro_reviewer_result="needs_facts",
        macro_reviewer_feedback="More data",
        facts_version=1
    )
    result = researcher_subgraph.run_researcher_subgraph_node(state, {})

    # Проверяем, что инкрементальная склейка работает даже при краше
    assert "--- ADDITIONAL RESEARCH DATA (PARTIAL) ---" in result["raw_facts"]
    assert "Экстренные новые факты" in result["raw_facts"]
    assert result["facts_version"] == 2

    called_messages = mock_llm.invoke.call_args[0][0]
    assert not isinstance(called_messages[-1], AIMessage)
    assert isinstance(called_messages[-1], ToolMessage)
