from config import settings
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Optional
from langchain_core.prompts import ChatPromptTemplate

class PlanSection(BaseModel):
    name: str = Field(description="Name of the section")
    description: str = Field(description="Detailed description of the section")
    page_count: float = Field(description="Page count of the section")

class PlannerResponse(BaseModel):
    reasoning : str = Field(
        description="Your reasoning about the plan. Here you can count different blocks' pages, so that they sum up to "
                    "the exact number needed. Finish this block only after you've completed drafting the plan "
                    "and rechecked that you're ready to write it.")

    plan : list[PlanSection] = Field(
        description="Field for the plan. List items should represent paragraphs.")


def main():

    paper_topic = input("Тема работы: ")
    page_count = int(input("Количество страниц: "))
    academic_level = input("Академический уровень: ")
    additional_instructions = input("Дополнительные инструкции: ")

    llm = ChatOpenAI(
        model=settings.llm.model_name,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}}
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
Role: You are an expert Academic Architect and Research Strategist. Your sole purpose is to design comprehensive, logical, and high-quality outlines for school and college reports (research papers). Your output serves as the foundational blueprint for downstream AI writing models.
Objective: Transform a given topic into a detailed, hierarchical outline that ensures academic rigor, flow, and depth.
Operational Guidelines:
    Logical Progression: Organize sections from general theory to specific analysis, ending with a synthesis of findings (Introduction -> Context/Literature Review -> Core Analysis -> Future Implications -> Conclusion).
    Granular Detail: Every main chapter must contain 2–4 specific sub-points. These sub-points must be descriptive enough to guide a writing model on exactly what arguments or data to include. For each sub-point, provide a brief one-sentence instruction on the specific focus for the writing model.
    Academic Standard: Use formal, scholarly terminology. Avoid vague titles like "Interesting Facts"; instead, use "Analysis of Key Empirical Data."
    Research-Driven: Ensure the outline includes sections for methodology, historical background, and current state-of-the-art perspectives where applicable.
    Structural Balance: Ensure that the weight of the outline is focused on the core analysis rather than over-extending on the introduction or history.
    The total length of the final paper should be exactly the number of pages the user told you. Each section has a page count field, decide how many pages each section needs and distribute the length among the sections. During the reasoning stage, sum up all the page count fields and recheck that the total number of pages meets the requirements.
Required Outline Components:
    Introduction: Must specify the background, problem statement, and objectives.
    Main Body: Clearly numbered chapters and sub-sections with distinct thematic focuses.
    Conclusion: Summary of findings, final synthesis, and suggestions for further study.
    Bibliography/References: A dedicated section for source attribution.
Tone & Style:
    Professional, objective, and analytical.
    Concise phrasing (no "fluff").
Output Format: JSON
"""),
        ("human", """
Please generate a detailed research paper outline for the following parameters:
- Topic: {paper_topic}
- Page Count: {page_count}
- Academic Level: {academic_level}
- Additional Context: {additional_instructions}

Ensure the logical flow follows: Introduction -> Literature Review -> Core Analysis -> Conclusion.
""")
    ])

    structured_llm = llm.with_structured_output(PlannerResponse, method="function_calling")

    planner_chain = prompt | structured_llm

    result = planner_chain.invoke({
        "paper_topic": paper_topic,
        "page_count": page_count,
        "academic_level": academic_level,
        "additional_instructions": additional_instructions,
    })

    print(result.reasoning)
    print(*result.plan, sep='\n')


if __name__ == "__main__":
    main()
