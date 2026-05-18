import asyncio
import logging
from app.workflow import run_research_graph


async def main():
    logging.basicConfig(level=logging.INFO)
    topic = input("Введите тему для исследования: ")

    result = await run_research_graph(topic, research_id="1")

    print(result)


if __name__ == "__main__":
    asyncio.run(main())
