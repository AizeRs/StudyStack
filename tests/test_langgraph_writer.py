from unittest.mock import patch, MagicMock
import pytest
from langchain_core.messages import AIMessage

from schema import ResearchPaperState
from writer_subgraph import run_writer_subgraph_node

@patch('writer_subgraph.llm')
def test_run_writer_subgraph_node(mock_llm: MagicMock):
    mock_llm.invoke.side_effect = [AIMessage("A perfect research paper text"),
                                   AIMessage("A second perfect research paper text"),
                                   AIMessage("An enhanced version of the perfect research paper text")]

    first_RP_state = ResearchPaperState(
        research_topic="topic",
        raw_facts="some facts",
        research_id=1)

    first_result = run_writer_subgraph_node(first_RP_state)

    assert first_result == {"main_paper_text": "A perfect research paper text"}
    mock_llm.invoke.assert_called_once()
    first_call_msgs = mock_llm.invoke.call_args_list[0].args[0]

    assert len(first_call_msgs) == 3

    #WIP:
    second_RP_state = ResearchPaperState(
        research_topic="topic",
        research_id=1,
        raw_facts="some facts + some other facts",
        facts_version=2)

    second_result = run_writer_subgraph_node(second_RP_state)

    assert second_result == {"main_paper_text": "A second perfect research paper text"}
    second_call_msgs = mock_llm.invoke.call_args_list[1].args[0]

    assert len(second_call_msgs) == 3
    assert "some other facts" in second_call_msgs[1].content

    third_RP_state = ResearchPaperState(
        research_topic="topic",
        research_id=1,
        facts_version=2,
        macro_reviewer_result="needs_rewrite",
        macro_reviewer_feedback="SOME_TEST_FEEDBACK")

    third_result = run_writer_subgraph_node(third_RP_state)

    assert third_result == {"main_paper_text": "An enhanced version of the perfect research paper text"}
    third_call_msgs = mock_llm.invoke.call_args_list[2].args[0]

    assert len(third_call_msgs) == 5 # System, human, human, AI, human (reviewer)
    assert "Критик" in third_call_msgs[4].content
