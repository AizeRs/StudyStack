import asyncio
import uuid

from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
import logging

from app.config import settings
from app.workflow import stream_research_graph

class Form(StatesGroup):
    waiting_for_topic = State()


# Инициализация бота и диспетчера
if settings.telegram is None:
    raise ValueError("Telegram bot token is required to run this bot, but is not set in .env file")


bot = Bot(token=settings.telegram.token.get_secret_value())
dp = Dispatcher()


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer("Привет! Это бот для сбора информации и написания отчётов по заданной теме.\n"
                         "Для начала работы используй команду /research")

# Хендлер на команду /research
@dp.message(Command("research"))
async def cmd_research(message: types.Message, state: FSMContext):
    await message.answer("Привет! Отправь мне тему для изучения и я отправлю отчёт.")
    await state.set_state(Form.waiting_for_topic)


# Хендлер, который сработает только если пользователь в состоянии waiting_for_topic
@dp.message(Form.waiting_for_topic)
async def do_research(message: types.Message, state: FSMContext):
    if message.text is None:
        await message.answer("На данный момент поддерживается только текстовый ввод темы.")
        return

    report_topic = message.text
    report_id = str(message.from_user.id) + uuid.uuid4().hex

    status_msg = await message.answer(f"Тема отчёта принята. Запускаю агентов... ⏳")
    final_text = ""

    async for node_name, update in stream_research_graph(report_topic, research_id=report_id):

        if node_name == "researcher":
            # Узел ресерчера отработал
            await status_msg.edit_text(
                f"✅ Факты собраны (версия {update.get('facts_version', 1)}). Передаю писателю...")

        elif node_name == "writer":
            # Узел писателя отработал
            await status_msg.edit_text("✍️ Черновик написан. Отправляю критику на проверку...")
            if "main_paper_text" in update:
                final_text = update["main_paper_text"]

        elif node_name == "macro_reviewer":
            # Узел критика отработал
            result = update.get("macro_reviewer_result")
            if result == "needs_rewrite":
                await status_msg.edit_text("❌ Критик забраковал текст. Отправляю на переписывание...")
            elif result == "needs_facts":
                await status_msg.edit_text("⚠️ Критику не хватило фактов. Ищем дополнительную информацию...")
            elif result == "approved":
                await status_msg.edit_text("🎉 Критик одобрил финальный текст!")

    if final_text:
        await message.answer(final_text)
    else:
        logging.error(f"ОШИБКА: Финальный текст не был сгенерирован!!! {report_id=}")


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
