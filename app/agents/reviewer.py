from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, SystemMessage

from app.config import llm

import logging
from app.schema import ResearchPaperState, ReviewerState, FirstReviewerResponse, NoFactsFirstReviewerResponse


def get_macro_reviewer_prompt(extra_research_forbidden: bool) -> str:
    facts_instruction = "" if extra_research_forbidden else "- If the `raw_facts` are too poor to write a good paper, demand more research ('needs_facts').\n"

    return f"""Role: You are a strict Academic Macro-Reviewer. 
Your objective is to evaluate a drafted paper against the gathered `raw_facts` and the original user task. You are the gatekeeper for factual accuracy and foundational structure.
Evaluation Rules:
1. Grounding is Absolute: The writer MUST NOT hallucinate. Every claim must be supported by the `raw_facts`.
2. Ignore Micro-Style: Do not focus on minor typos or word choice. Focus on data integration, logic, and completeness.
3. Strict Scoring: Be critical. A score of 5.0 means flawless execution. A score below 4.0 in any category requires a rewrite.
Workflow:
- Analyze the user Task, the `raw_facts`, and the current Draft.
- Generate your evaluation using the required structured output.
{facts_instruction}- If the writer ignored facts or structured the text poorly, demand a rewrite ('needs_rewrite') and list exactly what to fix.
- Approve ONLY if the foundation is solid ('approved').
"""


async def agent_node(state: ReviewerState, config: RunnableConfig) -> dict:
    extra_research_forbidden = config.get("configurable", {}).get("extra_research_forbidden", False)

    logging.info(f"[Reviewer Agent] Инициализация LLM. Строгий запрет на доресерч: {extra_research_forbidden}")

    if extra_research_forbidden:
        structured_llm = llm.with_structured_output(NoFactsFirstReviewerResponse, method="function_calling")
    else:
        structured_llm = llm.with_structured_output(FirstReviewerResponse, method="function_calling")

    logging.info("[Reviewer Agent] Запуск генерации оценки...")
    response: NoFactsFirstReviewerResponse | FirstReviewerResponse \
        = await structured_llm.ainvoke(state.messages)

    logging.info(f"[Reviewer Agent] Оценки: {response.scores}")
    logging.info(f"[Reviewer Agent] Вердикт: {response.review_result}")
    logging.info(f"[Reviewer Agent] Фидбек (превью): {response.feedback[:150]}...")

    return {"feedback": response.feedback, "scores": response.scores, "review_result": response.review_result}


workflow = StateGraph(ReviewerState)

workflow.add_node("agent", agent_node)

workflow.add_edge(START, "agent")
workflow.add_edge("agent", END)

checkpointer = MemorySaver()

app = workflow.compile(checkpointer=checkpointer)


async def run_macro_reviewer_subgraph_node(state: ResearchPaperState, config: RunnableConfig) -> dict:
    current_thread = f"reviewer_{state.research_id}_v{state.facts_version}"
    logging.info(f"СТАРТ СУБГРАФА: MACRO REVIEWER")
    logging.info(f"Ветка памяти (Thread ID): {current_thread}")
    logging.info(f"Запрет на доресерч (extra_research_forbidden): {state.extra_research_forbidden}")

    agent_config: RunnableConfig = {
        "configurable": {
            "thread_id": current_thread,
            "extra_research_forbidden": state.extra_research_forbidden
        },
    }

    last_iteration = config.get("configurable", {}).get("macro_reviewer_max_iterations", 3)

    match state.macro_reviewer_iteration:
        case 1:
            messages = [
                SystemMessage(get_macro_reviewer_prompt(state.extra_research_forbidden)),
                HumanMessage(f"---RAW FACTS---:\n{state.raw_facts}"),
                HumanMessage(f"---RESEARCH PAPER TEXT---:\n{state.main_paper_text}"),
                HumanMessage("Start your review now.")
            ]

        case val if val == last_iteration:
            messages = [
                HumanMessage(
                    f"ATTENTION: This is the final evaluation iteration.\n\n"
                    f"---FINAL DRAFT SUBMISSION---:\n{state.main_paper_text}\n\n"
                    f"Evaluate this final draft. Check specifically if the writer resolved the issues from your previous feedback:\n"
                    f"{state.macro_reviewer_feedback}\n\n"
                    f"Be decisive. If the foundation is now solid and meets the threshold, output 'approved'. "
                    f"If it still fails critically, output 'needs_rewrite'."
                )
            ]

        case _:
            messages = [
                HumanMessage(
                    f"The writer has submitted a revised draft based on your feedback.\n\n"
                    f"---UPDATED RESEARCH PAPER TEXT---:\n{state.main_paper_text}\n\n"
                    f"Task 1: Verify if the following previous issues were fully resolved:\n"
                    f"{state.macro_reviewer_feedback}\n\n"
                    f"Task 2: Scan the entire new draft for any OTHER critical macro-level flaws "
                    f"(hallucinations, logical breaks, or missing core facts from the raw_facts). "
                    f"Do not ignore new issues introduced during the rewrite or previously missed flaws."
                )
            ]


    input_data = {"messages": messages}

    result = await app.ainvoke(input_data, config=agent_config)
    pydantic_result = ReviewerState(**result)

    changes = {
        "macro_reviewer_result": pydantic_result.review_result,
        "macro_reviewer_feedback": pydantic_result.feedback,
        "macro_reviewer_scores": pydantic_result.scores,
        "macro_reviewer_iteration": state.macro_reviewer_iteration + 1
    }

    if pydantic_result.review_result != "needs_facts":
        logging.info(
            "[Reviewer Manager] Факты одобрены критиком. Блокируем возможность менять фундамент на будущих итерациях.")
        changes["extra_research_forbidden"] = True

    logging.info(f"ЗАВЕРШЕНИЕ СУБГРАФА: MACRO REVIEWER")

    return changes