from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from typing import Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, AIMessage
from langgraph.prebuilt import ToolNode

from config import settings
from search_tool_draft import search_web, read_webpage
from langchain_openai import ChatOpenAI
from langgraph.errors import GraphRecursionError
import logging


SYSTEM_PROMPT = """
Role: You are a Senior Research Specialist. Your mission is to conduct exhaustive investigations and gather high-quality "raw material" for a professional writer. You do not write the final narrative; instead, you provide the structural foundation and evidence required to create a masterpiece.
Available Tools:
search_web: Use this to identify key sources, news, academic papers, and diverse perspectives.
read_webpage: Use this to extract detailed information, data points, and direct quotes from specific URLs.
Core Objectives:
Depth over Breadth: Don't just skim the surface. Find the nuances, contradictions, and specific details that a general search would miss.
Direct Evidence: You must provide direct, verbatim quotes from credible sources.
Source Integrity: Always attribute information to a specific URL and title.
Data Categorization: Organize the gathered material into logical "buckets" (e.g., Historical Context, Key Stakeholders, Technical Specifications, Current Controversies).
Research Methodology & Workflow
Phase 1: Initial Discovery. Perform multiple search_web queries using different keywords to ensure a 360-degree view of the topic.
Phase 2: Deep Extraction. Use read_webpage on at least 2 high-quality sources. Look for specific statistics, expert opinions.
Phase 3: Synthesis. Group the information into a structured research dossier.
Output Format Requirements
Your final output must be a comprehensive document structured as follows:
Executive Summary of Findings: A brief overview of what the research uncovered.
Key Themes & Data Points:
Theme Title:
Detailed findings.
Direct Quotes: "Quote text here" — [Source Name/URL].
Data/Statistics: Hard numbers and facts.
Primary Sources & Citations: A list of all URLs visited with a one-sentence description of what each provides.
Suggested Angles for the Writer: Ideas on how the writer could frame this story based on the evidence found.
Operational Constraints
No Hallucinations: If you cannot find a specific fact, state that it was not found.
Verbatim Accuracy: Quotes must be 100% accurate to the source text.
Tone: Objective, analytical, and professional.
"""

class ResearcherState(BaseModel):
    messages: Annotated[list[AnyMessage], add_messages] = Field(default_factory=list)


my_tools = [search_web, read_webpage]
tool_node = ToolNode(my_tools)

llm = ChatOpenAI(
    base_url="http://localhost:5001/v1/",
    api_key="sk-no-key-needed",
    model="qwen3.5-2B",
    temperature=0.0
)


def agent_node(state: ResearcherState) -> dict:
    llm_with_tools = llm.bind_tools(my_tools)
    response = llm_with_tools.invoke(state.messages)

    return {"messages": [response]}


def should_continue(state: ResearcherState) -> str:
    last_message = state.messages[-1]

    if getattr(last_message, 'tool_calls', None):
        logging.info("Модель вызвала тул, передаём управление ему")
        return "continue"

    logging.info("Модель выдала ответ, завершаем цикл")
    return "end"

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - [%(levelname)s] - %(message)s"
    )


    workflow = StateGraph(ResearcherState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")

    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "continue": "tools",
            "end": END
        }
    )

    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()

    app = workflow.compile(checkpointer=checkpointer)

    config : RunnableConfig = {
        "configurable": {"thread_id": "user_1"},
        "recursion_limit": 15
    }

    task = input("Введите тему для исследования: ")

    try:
        logging.info("Начальный инференс")
        initial_input = {"messages": [HumanMessage(content=SYSTEM_PROMPT + f"TASK: {task}. Start your research now.")]}

        result = app.invoke(initial_input, config=config)

        final_pydantic_state = ResearcherState(**result)
        print("Финальный текст:\n", final_pydantic_state.messages[-1].content)

    except GraphRecursionError:
        logging.info("Достигнут лимит вызовов тулов. Начинаем graceful shutdown")

        crash_state = app.get_state(config)

        messages = crash_state.values.get("messages", [])

        if messages and isinstance(messages[-1], AIMessage):
            logging.info("Цикл оборвался до результата выполнения тула. Удаляем последний ответ модели и добавляем системное уведомление.")

        if messages and isinstance(messages[-1], ToolMessage):
            messages[-1].content += "\n\n[СИСТЕМНОЕ УВЕДОМЛЕНИЕ]: Лимит поиска исчерпан. Сформируй итоговый ответ на основе этих данных."

        else:
            print(f"Ошибка! Невозможно добавить системное уведомление. {len(messages)=} {type(messages[-1])=}")


if __name__ == "__main__":
    main()
