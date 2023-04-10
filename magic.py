import easyocr
import requests

CORR_KEY = 'bOzCJsRYPqLTPwvy'


def magic(filename):
    txt = easy_ocr_recognition(filename)
    res = correction(txt)
    return res


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

# Egor   |  |  |
#       v  v  v


def easy_ocr_recognition(file_path):
    reader = easyocr.Reader(["ru", "en"])
    result = reader.readtext(file_path, detail=0)
  
    string = ''
    for k in result:
        string += k + ' '

    return string