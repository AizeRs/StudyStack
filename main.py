"""
Точка входа для запуска приложения в консольном режиме (CLI).
Обеспечивает интерактивный сбор параметров исследования и 
потоковый вывод (streaming) статусов из графа LangGraph.
"""
import asyncio
from uuid import uuid4
import logging
import sys
from app.workflow import stream_research_graph

async def main():
    print("==================================================")
    print("     StudyStack: ИИ-генератор учебных работ       ")
    print("==================================================")
    
    # Интерактивный сбор данных через stdin
    topic = input("Введите тему для исследования: ").strip()
    if not topic:
        print("Тема не может быть пустой. Выход.")
        sys.exit(1)
        
    pages_str = input("Желаемое количество страниц (по умолчанию 3): ").strip()
    pages = int(pages_str) if pages_str.isdigit() else 3
    
    level = input("Укажите уровень работы (school/college/phd, по умолчанию 'college'): ").strip()
    if not level:
        level = "college"
        
    instructions = input("Дополнительные инструкции (опционально, нажмите Enter для пропуска): ").strip()
    
    output_file = input("Имя файла для сохранения (по умолчанию research_paper.md): ").strip()
    if not output_file:
        output_file = "research_paper.md"
        
    # Настройка минимального уровня логирования для чистоты консоли
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    research_id = uuid4().hex[:8]
    print(f"\n🚀 Инициализация исследования: '{topic}' (ID: {research_id})")
    print(f"📊 Параметры: {pages} стр., Уровень: {level}")
    
    final_text = ""

    try:
        print("\n⏳ Запуск конвейера. Пожалуйста, подождите...")
        print("-" * 50)
        
        # Запускаем граф в режиме стриминга, чтобы показывать прогресс
        async for event in stream_research_graph(
            research_topic=topic, 
            research_id=research_id, 
            research_length=pages, 
            academic_level=level, 
            additional_instructions=instructions
        ):
            event_type = event.get("type")
            
            if event_type == "error":
                print(f"❌ Ошибка пайплайна: {event.get('message')}")
                break
            elif event_type == "node_update":
                node_name = event.get("node_name")
                state_update = event.get("state_update", {})
                print(f"  ✅ Завершён шаг пайплайна: {node_name}")
                
                # Отлавливаем финальный результат
                if "main_paper_text" in state_update:
                    final_text = state_update["main_paper_text"]
            elif event_type == "custom":
                print(f"  ℹ️ {event.get('message')}")
            elif event_type == "chapter_status":
                phase = event.get("phase")
                idx = event.get("chapter_index")
                name = event.get("chapter_name")
                status = event.get("status")
                print(f"  ⏳ Глава {idx} [{name}] ({phase}): {status}")

        print("-" * 50)
        if final_text:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(final_text)
            print(f"🎉 [УСПЕХ] Документ успешно сгенерирован и сохранён в файл: {output_file}")
        else:
            print("❌ [ОШИБКА] Граф завершил работу, но финальный текст не был сгенерирован.")
            
    except KeyboardInterrupt:
        print("\n\n⚠️ Выполнение прервано пользователем.")
        sys.exit(0)
    except Exception as e:
        print(f"\n💥 [КРИТИЧЕСКАЯ ОШИБКА] Произошел сбой во время работы графа: {e}")
        sys.exit(1)

if __name__ == "__main__":
    # Фикс для Windows (чтобы избежать ошибки EventLoopClosed при завершении httpx/asyncio)
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
