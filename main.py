import logging

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, filters, MessageHandler, CallbackQueryHandler

import os
import glob

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
        mode[user] = 0
    keyboard = []
    message = ['', "Отправь мне картинку с текстом, и я его распознаю и исправлю!",
               "Отправь мне картинку с текстом, и я его распознаю!",
               "Отправь мне текст, и я его исправлю!"]
    if mode[user] == 0:
        text = 'Пожалуйста, выберите режим:'
        keyboard.append([InlineKeyboardButton("Распознать рукописный текст", callback_data=2)])
        keyboard.append([InlineKeyboardButton("Распознать и исправить текст ✅", callback_data=1)])
        keyboard.append([InlineKeyboardButton("Исправить печатный текст", callback_data=3)])
    else:
        text = message[mode[user]]
        keyboard.append([InlineKeyboardButton("Вернуться в меню", callback_data=0)])
    reply_markup = InlineKeyboardMarkup(keyboard)
    await context.bot.send_message(chat_id=user, text=text, reply_markup=reply_markup)


async def photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    if mode[user] == 3:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Вернуться в меню", callback_data=0)]])
        await context.bot.send_message(chat_id=user, text="Пожалуйста, отправьте печатный текст или смените режим!",
                                       reply_markup=keyboard)
    else:
        file_id = update.message.photo[-1].file_id
        new_file = await context.bot.get_file(file_id)
        filename = new_file.file_path.split('/')[-1]
        await new_file.download_to_drive(custom_path=f'img/{filename}')

        await context.bot.send_message(chat_id=user, text="⏳ Подождите, ваше изображение обрабатывается...")
        txt = magic(filename=f'img/{filename}', mode=mode.get(user, 1))
        print(txt)
        clear('./img')
        clear('./res')

        await context.bot.send_message(chat_id=update.effective_chat.id, text=txt, parse_mode='HTML')
        await start(update, context)


async def text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_chat.id
    if mode[user] != 3:
        keyboard = InlineKeyboardMarkup([[InlineKeyboardButton("Вернуться в меню", callback_data=0)]])
        await context.bot.send_message(chat_id=user, text="Пожалуйста, отправьте картинку или смените режим!",
                                       reply_markup=keyboard)
    else:
        txt = update.message.text
        res = magic(text=txt, mode=3)
        await context.bot.send_message(chat_id=user, text=res, parse_mode='HTML')


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

    text_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), text)
    application.add_handler(text_handler)

    application.add_handler(CallbackQueryHandler(button))

    application.run_polling()
