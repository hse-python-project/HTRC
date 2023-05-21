import easyocr
import requests
from correction import correct_text_final
from recognition import recognise

CORR_KEY = 'bOzCJsRYPqLTPwvy'


def magic_without_correction(filename):
    res = easy_ocr_recognition(filename)
    return res


def magic_with_correction(filename):
    txt = easy_ocr_recognition(filename)
    first_fix = correction(txt)
    result = correct_text_final(first_fix, "ru")  # добавить исправление языка текста на английский при необходимости
    text = result[0]
    mistake_counter = result[1]
    return (text, mistake_counter)


def correction(text):
    params = {'text': text, 'language': 'ru-RU', 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/grammar", params=params)
    grammar_mistakes = response.json()['response']['errors']
    params = {'text': text, 'language': 'ru-RU', 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/spelling", params=params)
    spelling_mistakes = response.json()['response']['errors']
    mistakes = grammar_mistakes + spelling_mistakes
    for mistake in mistakes:
        text = text[:mistake['offset']] + mistake['better'][0] + text[mistake['offset'] + mistake['length']:]
    return text


def magic(filename, mode):
    res = recognise(read_path=filename, draw_type='rect')
    return ' '.join(box['text'] for box in res['predictions'])


def easy_ocr_recognition(file_path):
    reader = easyocr.Reader(["ru", "en"])
    result = reader.readtext(file_path, detail=0)

    string = ''
    for k in result:
        string += k + ' '

    return string
