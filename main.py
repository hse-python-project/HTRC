import logging

from telegram import Update
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler

from magic import magic

TOKEN = '5631958925:AAGBOxvkn3JTiR2dUAJ59_IL7qMEnNycLOM'

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Отправь мне картинку с текстом!")


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    file_id = update.message.photo[-1].file_id
    new_file = await context.bot.get_file(file_id)
    filename = new_file.file_path.split('/')[-1]
    await new_file.download_to_drive(custom_path=f'img/{filename}')

    '''with open(f'res/{filename}', 'wb') as res:
        res.write(magic(f'img/{filename}'))*/'''

    txt = magic(f'img/{filename}')

    await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)


if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    photo_handler = MessageHandler(filters.PHOTO & (~filters.COMMAND), photo)
    application.add_handler(photo_handler)

    application.run_polling()
