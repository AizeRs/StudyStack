import logging
from app.workflow import run_research_graph

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    topic = input("Введите тему для исследования: ")
    result = run_research_graph(topic, research_id=1)
    print(result)
