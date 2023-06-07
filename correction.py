import requests
from langdetect import detect
import ai21

ai21.api_key = 'ixlBVyHgdogp531oVryy4uw0hubfWAnf'


def english_correction(text):
    response = ai21.GEC.execute(text=text)
    corrected_text = text
    corrections = response["corrections"]
    for curr_correction in reversed(corrections):
        corrected_text = corrected_text[:curr_correction["startIndex"]] + "<i>" + curr_correction[
            'suggestion'] + "</i>" + corrected_text[curr_correction["endIndex"]:]
    return corrected_text


CORR_KEY = 'bOzCJsRYPqLTPwvy'


def correction_with_punctuation(text):
    language = "ru-RU" if detect(text) == "ru" else "en-GB"

    if language == "en-GB":
        return english_correction(text)

    grammar_corrected_text = ""

    for it in range(2):
        params = {'text': text, 'language': language, 'ai': 0, 'key': CORR_KEY}
        response = requests.get(url="https://api.textgears.com/grammar", params=params)
        grammar_mistakes = response.json()['response']['errors']

        grammar_corrected_text = ""
        i = 0
        for mistake in grammar_mistakes:
            if len(mistake['better']) == 0:
                continue
            grammar_corrected_text += text[i:mistake['offset']] + '<i>' + mistake['better'][0] + '</i>'
            i += len(text[i:mistake['offset']]) + mistake['length']

        grammar_corrected_text += text[i:]
        text = (grammar_corrected_text + '.')[:-1]

    params = {'text': grammar_corrected_text, 'language': language, 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/spelling", params=params)
    spelling_mistakes = response.json()['response']['errors']

    spelling_corrected_text = ""
    i = 0
    for mistake in spelling_mistakes:
        spelling_corrected_text += grammar_corrected_text[i:mistake['offset']] + '<i>' + mistake['better'][0] + '</i>'
        i += len(grammar_corrected_text[i:mistake['offset']]) + mistake['length']

    spelling_corrected_text += grammar_corrected_text[i:]
    return grammar_corrected_text


def main():
    txt = "Thhhis exxample for text corection which is, howewer dificult task."
    txt1 = 'Это прример для того чтобы посмотреть как рааботает програма основанная на искусственном интелекте встроенном в компьютер. Однакко не стоит забывать что ИИ это новая техналогия которая пока не очень надежная'
    print(correction_with_punctuation(txt1))


if __name__ == '__main__':
    main()
