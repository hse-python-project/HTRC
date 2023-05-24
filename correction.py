from spellchecker import SpellChecker
import requests
from langdetect import detect

CORR_KEY = 'bOzCJsRYPqLTPwvy'


def correction_with_punctuality(text):
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


def correct_text_final(text):
    speller = SpellChecker(language=detect(text))
    old_text_version = text.split()
    grammar_fixed_text = []
    punctual_symbols = [',', '.', '?', '-', '!', '...', ';', ':']
    mistake_counter = 0
    for word in text.split():
        if word in punctual_symbols:
            grammar_fixed_text.append((word, 0))
        if speller.correction(word) is None:
            grammar_fixed_text.append((word, 0))
        else:
            grammar_fixed_text.append((speller.correction(word), 1))
            mistake_counter += 1
    return (grammar_fixed_text, mistake_counter)
