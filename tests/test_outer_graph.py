import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage

from app.schema import ResearchPaperState
from app.config import settings
from app import workflow

ft, st = settings.reviewers.macro_reviewer_first_threshold, settings.reviewers.macro_reviewer_second_threshold
last_iter: int = settings.reviewers.macro_reviewer_max_iterations

testdata_macro_reviewer_router = [
    (2, "approved", [4.0, 4.0, 4.0, 4.0, 4.0], "feedback", "end"),
    (2, "needs_facts", [4.0, 4.0, 4.0, 4.0, 4.0], "feedback", "researcher"),
    (2, "needs_rewrite", [st, st, st, st, st], "feedback", "writer"), # no threshold exit
    (2, "needs_rewrite", [ft, ft, ft, ft, ft], "feedback", "end"), # first score threshold exit
    (3, "needs_rewrite", [st, st, st, st, st], "feedback", "end"), # second score threshold exit
    (last_iter + 1, "needs_rewrite", [0, 0, 0, 0, 0], "critical feedback", "end"), # iterations limit exit
]
ids = ["basic end", "needs_facts leads to researcher", "needs_rewrite without threshold exit",
       "first score threshold exit", "second score threshold exit", "iterations limit exit"]


@pytest.mark.parametrize("iteration,result,scores,feedback,expected", testdata_macro_reviewer_router, ids=ids)
def test_macro_reviewer_router(iteration, result, scores, feedback, expected):
    state = ResearchPaperState(
        macro_reviewer_iteration=iteration,
        research_topic="topic",
        research_id=1,
        macro_reviewer_result=result,
        macro_reviewer_scores=scores,
        macro_reviewer_feedback=feedback)

    assert workflow.macro_reviewer_router(state) == expected


@patch('app.agents.reviewer.app.invoke')
@patch('app.agents.writer.app.invoke')
@patch('app.agents.researcher.app.invoke')
def test_run_research_graph_straight_approval(mock_researcher, mock_writer, mock_reviewer):
    # Идеальный проход с первой итерации
    mock_researcher.return_value = {"messages": [AIMessage(content="Facts from researcher")]}
    mock_writer.return_value = {"messages": [AIMessage(content="Perfect first draft")]}
    mock_reviewer.return_value = {
        "review_result": "approved",
        "feedback": "feedback",
        "scores": [5.0, 5.0, 5.0, 5.0, 5.0]
    }

    result = workflow.run_research_graph("Test Topic", research_id=201)

    assert result == "Perfect first draft"

    mock_researcher.assert_called_once()
    mock_writer.assert_called_once()
    mock_reviewer.assert_called_once()


@patch('app.agents.reviewer.app.invoke')
@patch('app.agents.writer.app.invoke')
@patch('app.agents.researcher.app.invoke')
def test_run_research_graph_rewrite_loop(mock_researcher, mock_writer, mock_reviewer):
    # Ревьювер бракует текст, писатель переписывает
    mock_researcher.return_value = {"messages": [AIMessage(content="Facts")]}

    # Писатель вызывается дважды
    mock_writer.side_effect = [
        {"messages": [AIMessage(content="Bad draft")]},
        {"messages": [AIMessage(content="Fixed draft")]}
    ]

    # Ревьювер сначала отправляет на рерайт, потом одобряет
    mock_reviewer.side_effect = [
        {
            "review_result": "needs_rewrite",
            "feedback": "feedback",
            "scores": [3.0, 3.0, 3.0, 3.0, 3.0]
        },
        {
            "review_result": "approved",
            "feedback": "feedback",
            "scores": [4.5, 4.5, 4.5, 4.5, 4.5]
        }
    ]

    result = workflow.run_research_graph("Test Topic", research_id=202)

    assert result == "Fixed draft"
    mock_researcher.assert_called_once()  # Ресерчер не перевызывается
    assert mock_writer.call_count == 2
    assert mock_reviewer.call_count == 2


@patch('app.agents.reviewer.app.invoke')
@patch('app.agents.writer.app.invoke')
@patch('app.agents.researcher.app.invoke')
def test_run_research_graph_needs_facts_loop(mock_researcher, mock_writer, mock_reviewer):
    # Ревьювер требует больше фактов, графа возвращается к ресерчеру
    mock_researcher.side_effect = [
        {"messages": [AIMessage(content="Initial facts")]},
        {"messages": [AIMessage(content="Extended facts")]}
    ]

    mock_writer.side_effect = [
        {"messages": [AIMessage(content="Draft missing info")]},
        {"messages": [AIMessage(content="Draft with all info")]}
    ]

    mock_reviewer.side_effect = [
        {
            "review_result": "needs_facts",
            "feedback": "feedback",
            "scores": [4.0, 4.0, 4.0, 4.0, 4.0]
        },
        {
            "review_result": "approved",
            "feedback": "feedback",
            "scores": [4.5, 4.5, 4.5, 4.5, 4.5]
        }
    ]

    result = workflow.run_research_graph("Test Topic", research_id=203)

    assert result == "Draft with all info"
    assert mock_researcher.call_count == 2
    assert mock_writer.call_count == 2
    assert mock_reviewer.call_count == 2
