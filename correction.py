import requests
from spellchecker import SpellChecker
from langdetect import detect
import ai21

ai21.api_key = 'ixlBVyHgdogp531oVryy4uw0hubfWAnf'


def additional_correction(text):
    response = ai21.GEC.execute(text=text)
    # Use the corrections to fix the sentence
    corrected_text = text
    corrections = response["corrections"]
    for curr_correction in reversed(corrections):
        corrected_text = corrected_text[:curr_correction["startIndex"]] + curr_correction[
            'suggestion'] + corrected_text[curr_correction["endIndex"]:]

    return corrected_text


CORR_KEY = 'bOzCJsRYPqLTPwvy'


def correction_with_punctuation_old(text):
    language = "ru-RU" if detect(text) == "ru" else "en-GB"

    params = {'text': text, 'language': language, 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/grammar", params=params)
    grammar_mistakes = response.json()['response']['errors']
    params = {'text': text, 'language': language, 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/spelling", params=params)
    spelling_mistakes = response.json()['response']['errors']
    mistakes = grammar_mistakes + spelling_mistakes
    for mistake in mistakes:
        text = text[:mistake['offset']] + mistake['better'][0] + text[mistake['offset'] + mistake['length']:]
    return text


def correction_with_punctuation(text):
    language = "ru-RU" if detect(text) == "ru" else "en-GB"

    params = {'text': text, 'language': language, 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/spelling", params=params)
    spelling_mistakes = response.json()['response']['errors']

    # print(spelling_mistakes)

    print("Initial text:", text)

    spelling_corrected_text = ""
    i = 0
    for mistake in spelling_mistakes:
        # print('"' + spelling_corrected_text + '" -> "', end='')
        spelling_corrected_text += text[i:mistake['offset']] + '__' + mistake['better'][0] + '__'
        # print(spelling_corrected_text + '", copied everything from char ' + str(i) + " to char ", end='')
        i += len(text[i:mistake['offset']]) + mistake['length']
        # print(i)

    spelling_corrected_text += text[i:]
    # print("After correcting spelling:", spelling_corrected_text)

    grammar_corrected_text = ""

    for it in range(2):
        # print("-" * 20)
        # print("ITERATION", it)
        # print("CURRENT TEXT:", spelling_corrected_text)

        params = {'text': spelling_corrected_text, 'language': language, 'ai': 0, 'key': CORR_KEY}
        response = requests.get(url="https://api.textgears.com/grammar", params=params)
        grammar_mistakes = response.json()['response']['errors']

        # print(grammar_mistakes)
        grammar_corrected_text = ""
        i = 0
        for mistake in grammar_mistakes:
            if len(mistake['better']) == 0:
                continue
            # print('"' + grammar_corrected_text + '" -> "', end='')
            grammar_corrected_text += spelling_corrected_text[i:mistake['offset']] + '__' + mistake['better'][0] + '__'
            # print(grammar_corrected_text + '", copied everything from char ' + str(i) + " to char ", end='')
            i += len(spelling_corrected_text[i:mistake['offset']]) + mistake['length']
            # print(i)

        grammar_corrected_text += spelling_corrected_text[i:]

        # print("After correcting grammar:", grammar_corrected_text)

        spelling_corrected_text = (grammar_corrected_text + '.')[:-1]

    return grammar_corrected_text


def mark_corrected(grammar_fixed_text):
    ans = []
    for word, corrected in grammar_fixed_text:
        ans.append('**' + word + '**' if corrected else word)
    return ans


def correct_text_final(text):
    speller = SpellChecker(language=detect(text))
    old_text_version = text.split()
    grammar_fixed_text = []
    punctual_symbols = [',', '.', '?', '-', '!', '...', ';', ':']
    mistake_counter = 0
    res = ''
    print(text.split())
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
    teext = mark_corrected(grammar_fixed_text)
    for k in teext:
        res += k + " "
    return res


def main():
    txt = "Thhhis exxample for text corection which is, howewer dificult task."
    txt1 = 'Это прример для того чтобы посмотреть как рааботает програма основанная на искусственном интелекте встроенном в компьютер. Однакко не стоит забывать что ИИ это новая техналогия которая пока не очень надежная'
    # print(correction_with_punctuation(txt1))
    # print("-" * 20)
    print(additional_correction(correction_with_punctuation(txt)))
    # print(correction_with_punctuation_old(txt))
    # print(correct_text_final(txt))


if __name__ == '__main__':
    main()
