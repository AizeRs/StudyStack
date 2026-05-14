from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import SystemMessage, HumanMessage
from schema import ResearchPaperState, FirstReviewerResponse, NoFactsFirstReviewerResponse
import reviewer_subgraph


@patch('reviewer_subgraph.llm')
def test_run_macro_reviewer_iteration_1_rewrite(mock_llm: MagicMock):
    # Настройка мока для структурированного вывода
    mock_structured_llm = MagicMock()
    fake_response = FirstReviewerResponse(
        feedback="Missing some key points about pizza history.",
        scores=[3.5, 4.0, 4.0, 4.5, 4.0],
        review_result="needs_rewrite"
    )
    mock_structured_llm.invoke.return_value = fake_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    state = ResearchPaperState(
        research_topic="pizza history",
        research_id=1,
        raw_facts="Facts about pizza.",
        main_paper_text="Draft text.",
        macro_reviewer_iteration=1,
        extra_research_forbidden=False,
        facts_version=1
    )

    result = reviewer_subgraph.run_macro_reviewer_subgraph_node(state, {})

    assert result["macro_reviewer_result"] == "needs_rewrite"
    assert result["macro_reviewer_iteration"] == 2
    assert result["extra_research_forbidden"] is True

    # Проверка промптов первой итерации
    sent_messages = mock_structured_llm.invoke.call_args[0][0]
    assert isinstance(sent_messages[0], SystemMessage)
    assert "facts" in sent_messages[1].content.lower()


@patch('reviewer_subgraph.llm')
def test_run_macro_reviewer_needs_facts(mock_llm: MagicMock):
    mock_structured_llm = MagicMock()
    fake_response = FirstReviewerResponse(
        feedback="We need more data on crust types.",
        scores=[2.0, 3.0, 3.0, 4.0, 4.0],
        review_result="needs_facts"
    )
    mock_structured_llm.invoke.return_value = fake_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    state = ResearchPaperState(
        research_topic="pizza",
        research_id=2,
        raw_facts="Facts.",
        main_paper_text="Text.",
        macro_reviewer_iteration=1,
        extra_research_forbidden=False
    )

    result = reviewer_subgraph.run_macro_reviewer_subgraph_node(state, {})

    assert result["macro_reviewer_result"] == "needs_facts"


@patch('reviewer_subgraph.llm')
def test_run_macro_reviewer_intermediate_iteration(mock_llm: MagicMock):
    mock_structured_llm = MagicMock()
    fake_response = NoFactsFirstReviewerResponse(
        feedback="Better, but still missing logic.",
        scores=[4.0, 4.0, 4.0, 3.0, 4.0],
        review_result="needs_rewrite"
    )
    mock_structured_llm.invoke.return_value = fake_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    state = ResearchPaperState(
        research_topic="pizza",
        research_id=3,
        raw_facts="Facts.",
        main_paper_text="Updated text.",
        macro_reviewer_iteration=2,
        extra_research_forbidden=True,
        macro_reviewer_feedback="Previous feedback."
    )

    reviewer_subgraph.run_macro_reviewer_subgraph_node(state, {})

    # Проверка промежуточной итерации
    sent_messages = mock_structured_llm.invoke.call_args[0][0]
    assert "The writer has submitted a revised draft" in sent_messages[0].content
    assert "Previous feedback." in sent_messages[0].content

    # Проверка использования схемы без возможности выбора 'needs_facts'
    mock_llm.with_structured_output.assert_called_with(NoFactsFirstReviewerResponse, method="function_calling")


@patch('reviewer_subgraph.llm')
def test_run_macro_reviewer_final_iteration(mock_llm: MagicMock):
    mock_structured_llm = MagicMock()
    fake_response = NoFactsFirstReviewerResponse(
        feedback="Accepted.",
        scores=[5.0, 5.0, 5.0, 5.0, 5.0],
        review_result="approved"
    )
    mock_structured_llm.invoke.return_value = fake_response
    mock_llm.with_structured_output.return_value = mock_structured_llm

    # По умолчанию last_iteration = 3
    state = ResearchPaperState(
        research_topic="pizza",
        research_id=4,
        raw_facts="Facts.",
        main_paper_text="Final text.",
        macro_reviewer_iteration=3,
        extra_research_forbidden=True,
        macro_reviewer_feedback="Fix final summary."
    )

    result = reviewer_subgraph.run_macro_reviewer_subgraph_node(state, {})

    assert result["macro_reviewer_result"] == "approved"

    # Проверка ветки last_iteration
    sent_messages = mock_structured_llm.invoke.call_args[0][0]
    assert "final" in sent_messages[0].content
