import asyncio
from uuid import uuid4

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import logging

from app.config import settings
from app.workflow import stream_research_graph

from aiogram.types import BufferedInputFile, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class Form(StatesGroup):
    waiting_for_topic = State()
    waiting_for_length = State()
    waiting_for_academic_level = State()
    waiting_for_additional_instructions = State()


# Инициализация бота и диспетчера
if settings.telegram is None:
    raise ValueError("Необходим токен Telegram-бота, но он не задан в конфигурации (.env).")


bot = Bot(token=settings.telegram.token.get_secret_value())
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Это бот для сбора информации и написания отчётов по заданной теме.\n"
                         "Для начала работы используй команду /research")

# Хендлер на команду /research
@dp.message(Command("research"))
async def cmd_research(message: types.Message, state: FSMContext):
    await message.answer("Привет! Отправь мне тему для изучения (например: 'Влияние ИИ на образование').")
    await state.set_state(Form.waiting_for_topic)

@dp.message(Form.waiting_for_topic)
async def process_topic(message: types.Message, state: FSMContext):
    if not message.text:
        await message.answer("Пожалуйста, отправь текст.")
        return
    await state.update_data(topic=message.text)
    await message.answer("Сколько страниц примерно должен быть отчёт? (введи число, например 3)")
    await state.set_state(Form.waiting_for_length)

@dp.message(Form.waiting_for_length)
async def process_length(message: types.Message, state: FSMContext):
    if not message.text or not message.text.isdigit():
        await message.answer("Пожалуйста, введи число.")
        return
    await state.update_data(length=int(message.text))
    
    kb = ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="Школьник"), KeyboardButton(text="Студент")],
        [KeyboardButton(text="Аспирант/Профессор")]
    ], resize_keyboard=True)
    
    await message.answer("Какой академический уровень использовать?", reply_markup=kb)
    await state.set_state(Form.waiting_for_academic_level)

@dp.message(Form.waiting_for_academic_level)
async def process_academic_level(message: types.Message, state: FSMContext):
    level_mapping = {
        "Школьник": "high_school",
        "Студент": "college",
        "Аспирант/Профессор": "phd"
    }
    level = level_mapping.get(message.text, "college")
    
    await state.update_data(academic_level=level)
    await message.answer("Дополнительные инструкции (напишите текст или отправьте '-', чтобы пропустить):", reply_markup=ReplyKeyboardRemove())
    await state.set_state(Form.waiting_for_additional_instructions)

@dp.message(Form.waiting_for_additional_instructions)
async def do_research(message: types.Message, state: FSMContext):
    if not message.text:
        await message.answer("Пожалуйста, отправь текст.")
        return
        
    instructions = message.text if message.text.strip() != "-" else ""
    user_data = await state.get_data()
    
    report_topic = user_data['topic']
    research_length = user_data['length']
    academic_level = user_data['academic_level']
    
    report_id = str(message.from_user.id) + uuid4().hex

    status_msg = await message.answer("Тема отчёта принята. Запускаю агентов... ⏳")
    await asyncio.sleep(1.5)
    last_status_text = "🧭 Формирую структуру работы (план)..."
    await status_msg.edit_text(last_status_text)
    
    final_text = ""
    
    # Очищаем состояние
    await state.clear()
    
    total_chapters = 0
    
    skeleton_data = {}
    draft_data = {}

    async def update_status(new_text: str):
        nonlocal last_status_text
        if new_text != last_status_text:
            try:
                await status_msg.edit_text(new_text)
                last_status_text = new_text
            except Exception as ex:
                if "is not modified" not in str(ex).lower():
                    logging.warning(f"Ошибка обновления статуса: {ex}")

    try:
        async for event in stream_research_graph(
            report_topic, 
            research_id=report_id,
            research_length=research_length,
            academic_level=academic_level,
            additional_instructions=instructions
        ):
            event_type = event.get("type")
            
            if event_type == "error":
                await update_status(f"❌ Произошла критическая ошибка: {event.get('message')}")
                return
                
            if event_type == "chapter_status":
                phase = event.get("phase")
                idx = event.get("chapter_index")
                name = event.get("chapter_name")
                status = event.get("status")
                
                if phase == "skeleton":
                    for k in skeleton_data:
                        skeleton_data[k]["status"] = "✅ Готово"
                    skeleton_data[idx] = {"name": name, "status": status}
                    
                    lines = ["✍️ Сценарист продумывает структуру:"]
                    for k in sorted(skeleton_data.keys()):
                        lines.append(f"Глава {k}: {skeleton_data[k]['name']} - {skeleton_data[k]['status']}")
                    await update_status("\n".join(lines))
                    
                elif phase == "drafting":
                    draft_data[idx] = {"name": name, "status": status}
                    lines = ["🚀 Параллельная генерация глав:"]
                    for k in sorted(draft_data.keys()):
                        lines.append(f"Глава {k}: {draft_data[k]['name']} - {draft_data[k]['status']}")
                    await update_status("\n".join(lines))
                continue
                
            if event_type == "custom":
                msg = event.get("message")
                if msg:
                    await update_status(msg)
                continue
                
            if event_type == "node_update":
                node_name = event.get("node_name")
                update = event.get("state_update", {})

                if node_name == "planner":
                    total_chapters = len(update.get("macro_plan", []))
                    await update_status(f"📝 План составлен! Всего глав: {total_chapters}. Ищу источники... 🔍")

                elif node_name == "source_finder":
                    await update_status("🌐 Источники найдены. Запускаю параллельный анализ... 🧠")

                elif node_name == "scriptor":
                    await update_status("✍️ Скелет работы написан. Начинаю параллельную генерацию глав... 🚀")

                elif node_name == "final_concatenator":
                    await update_status("🎉 Все главы написаны и склеены! Подготавливаю файл... 📄")
                    if "main_paper_text" in update:
                        final_text = update["main_paper_text"]
    except Exception as e:
        await update_status(f"❌ Произошла непредвиденная ошибка: {str(e)}")
        logging.error(f"Ошибка в боте: {e}", exc_info=True)
        return

    if final_text:
        file_bytes = final_text.encode('utf-8')
        document = BufferedInputFile(file_bytes, filename=f"report_{report_id[:8]}.md")
        await message.answer_document(document, caption="Вот ваш готовый отчёт!")
        await status_msg.delete()
    else:
        logging.error(f"ОШИБКА: Финальный текст не был сгенерирован!!! {report_id=}")
        await status_msg.edit_text("❌ ОШИБКА: Финальный текст не был сгенерирован (сбой пайплайна).")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
