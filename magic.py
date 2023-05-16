import easyocr
import requests
import pprint

from recognition import recognise

CORR_KEY = 'bOzCJsRYPqLTPwvy'


def magic_without_correction(filename):
    res = easy_ocr_recognition(filename)
    return res


def magic_with_correction(filename):
    txt = easy_ocr_recognition(filename)
    res = correction(txt)
    return res


def magic_magic_recognition(filename):
    return recognise(read_path=filename, draw_type='rect')


def magic(filename, mode):
    res = magic_magic_recognition(filename=filename)
    print(res)
    return ' '.join(i['text'] for i in res['predictions'])
    # return magic_with_correction(filename) if mode == 1 else magic_without_correction(filename)


def correction(text):
    params = {'text': text, 'language': 'ru-RU', 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/grammar", params=params)
    # pprint(response.json())
    grammar_mistakes = response.json()['response']['errors']
    params = {'text': text, 'language': 'ru-RU', 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/spelling", params=params)
    spelling_mistakes = response.json()['response']['errors']
    mistakes = grammar_mistakes + spelling_mistakes
    for mistake in mistakes:
        text = text[:mistake['offset']] + mistake['better'][0] + text[mistake['offset'] + mistake['length']:]
    return text


def easy_ocr_recognition(file_path):
    reader = easyocr.Reader(["ru", "en"])
    result = reader.readtext(file_path, detail=0)
  
    string = ''
    for k in result:
        string += k + ' '

    return string
