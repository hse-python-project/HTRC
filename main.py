import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler, CallbackQueryHandler

from magic import magic

TOKEN = '5631958925:AAGBOxvkn3JTiR2dUAJ59_IL7qMEnNycLOM'

mode = {}

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    if user not in mode.keys():
        mode[user] = 1
    keyboard = []
    if mode[user] == 1:
        text = "Отправь мне картинку с текстом, и я его исправлю!"
        keyboard.append([InlineKeyboardButton("Лучше не исправляй", callback_data=2)])
    else:
        text = "Отправь мне картинку с текстом, и я его не исправлю!"
        keyboard.append([InlineKeyboardButton("Лучше исправь", callback_data=1)])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=user, text=text, reply_markup=reply_markup)


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    file_id = update.message.photo[-1].file_id
    new_file = await context.bot.get_file(file_id)
    filename = new_file.file_path.split('/')[-1]
    await new_file.download_to_drive(custom_path=f'img/{filename}')

    '''with open(f'res/{filename}', 'wb') as res:
        res.write(magic(f'img/{filename}'))*/'''

    txt = magic(f'img/{filename}', mode[user])

    await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    query = update.callback_query

    await query.answer()
    print(mode[user], query.data)

    mode[user] = int(query.data)
    await start(update, context)


if __name__ == '__main__':
    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    photo_handler = MessageHandler(filters.PHOTO & (~filters.COMMAND), photo)
    application.add_handler(photo_handler)

    application.add_handler(CallbackQueryHandler(button))

    application.run_polling()
