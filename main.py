import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler, CallbackQueryHandler

import os
import glob

import csv
import atexit

from magic import magic

TOKEN = '5631958925:AAGBOxvkn3JTiR2dUAJ59_IL7qMEnNycLOM'

mode = {}


def save_upon_exit():
    """with open('modes.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=';', quotechar='"')
        for user in mode:
            writer.writerow([user, mode[user]])"""
    return


logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)


def clear(path):
    files = glob.glob(f'{path}/*')
    for f in files:
        os.remove(f)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    if user not in mode.keys():
        mode[user] = 1
    keyboard = []
    if mode[user] == 1:
        text = "Отправь мне картинку с текстом, и я его распознаю и исправлю!"
        keyboard.append([InlineKeyboardButton("Выключить исправления ❌", callback_data=2)])
    else:
        text = "Отправь мне картинку с текстом, и я его распознаю!"
        keyboard.append([InlineKeyboardButton("Включить исправления ✅", callback_data=1)])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=user, text=text, reply_markup=reply_markup)


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    file_id = update.message.photo[-1].file_id
    new_file = await context.bot.get_file(file_id)
    filename = new_file.file_path.split('/')[-1]
    await new_file.download_to_drive(custom_path=f'img/{filename}')

    await context.bot.send_message(chat_id=update.effective_chat.id, text="Подождите, ваше изображение обрабатывается.")
    txt = magic(f'img/{filename}', mode[user])
    print(txt)
    clear('./img')
    clear('./res')

    await context.bot.send_message(chat_id=update.effective_chat.id, text=txt)
    await start(update, context)


async def button(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    query = update.callback_query

    await query.answer()

    mode[user] = int(query.data)
    await start(update, context)


if __name__ == '__main__':
    '''with open('mode.csv', encoding="utf8") as file:
        reader = csv.reader(file, delimiter=';', quotechar='"')
    print(reader)
    atexit.register(save_upon_exit)'''

    application = ApplicationBuilder().token(TOKEN).build()

    start_handler = CommandHandler('start', start)
    application.add_handler(start_handler)

    photo_handler = MessageHandler(filters.PHOTO & (~filters.COMMAND), photo)
    application.add_handler(photo_handler)

    application.add_handler(CallbackQueryHandler(button))

    application.run_polling()
