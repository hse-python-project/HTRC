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

def attention(grammar_fixed_text):
    ans = []
    for i in range(len(grammar_fixed_text)):
        if grammar_fixed_text[i][1] == 1:
            ans.append('**' + grammar_fixed_text[i][0] + '**')
        else:
            ans.append(grammar_fixed_text[i][0])
    return ans

def correct_text_final(text):
    speller = SpellChecker(language=detect(text))
    old_text_version = text.split()
    grammar_fixed_text = []
    punctual_symbols = [',', '.', '?', '-', '!', '...', ';', ':']
    mistake_counter = 0
    res = ''
    for word in text.split():
        if word in punctual_symbols:
            grammar_fixed_text.append((word, 0))
        if speller.correction(word) is None:
            grammar_fixed_text.append((word, 0))
        else:
            temp = speller.correction(word)
            if word == temp:
                grammar_fixed_text.append((temp, 0))
            else:
                grammar_fixed_text.append((speller.correction(word), 1))
                mistake_counter += 1
    teext = attention(grammar_fixed_text)
    for k in teext:
        res += k + " "
    return res
   
    
    
#tests
txt = "Thhis exxample for text corection"
txt1 = 'Это прример для того, чтобы посмотреть как аботает програма'
correction_with_punctuality(txt)
print(correct_text_final(txt))
