import requests
import ai21
from langdetect import detect

ai21.api_key = 'ixlBVyHgdogp531oVryy4uw0hubfWAnf'
CORR_KEY = 'bOzCJsRYPqLTPwvy'


def english_correction(text):
    """Finds and corrects mistakes in a given text in English."""
    
    response = ai21.GEC.execute(text=text)
    corrected_text = text
    corrections = response["corrections"]
    for curr_correction in reversed(corrections):
        corrected_text = corrected_text[:curr_correction["startIndex"]] + "<i>" + curr_correction[
            'suggestion'] + "</i>" + corrected_text[curr_correction["endIndex"]:]
    return corrected_text


def remove_extra_spaces(text):
    """Removes extra spaces, which appeared after adding HTML tags."""

    text = text.replace(' <i> ', '<i>').replace(' </i> ', '</i>')
    return text


def correct_mistakes(text, mistakes):
    """Corrects a list of mistakes in a given text."""

    mistakes = sorted(mistakes, key=lambda x: x['offset'], reverse=True)
    corrected_text = text
    for mistake in mistakes:
        if not mistake['better'] or mistake['type'] == 'duplication':
            continue
        corrected_text = corrected_text[:mistake["offset"]] + " <i> " + mistake[
            'better'][0] + " </i> " + corrected_text[mistake['offset'] + mistake['length']:]
    return corrected_text


def correction(text):
    """Finds and corrects mistakes in a given text."""

    language = "ru-RU" if detect(text) == "ru" else "en-GB"

    if language == "en-GB":
        return english_correction(text)

    params = {'text': text, 'language': language, 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/grammar", params=params)
    grammar_mistakes = response.json()['response']['errors']

    text = correct_mistakes(text, grammar_mistakes)

    params = {'text': text, 'language': language, 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/spelling", params=params)
    spelling_mistakes = response.json()['response']['errors']

    text = correct_mistakes(text, spelling_mistakes)

    return remove_extra_spaces(text)


def yandex_corr(txt):
    """Finds and corrects mistakes ia a given text with YandexSpeller API"""
    language = "ru-RU" if detect(txt) == "ru" else "en-GB"
    if language == "en-GB":
        return english_correction(txt)

    req = f"https://speller.yandex.net/services/spellservice.json/checkTexts?" + 'text=' + txt
    response = requests.get(req).json()[0]
    mistakes = []
    for mistake in response:
        mistakes.append({
            'offset': mistake['pos'],
            'length': mistake['len'],
            'better': mistake['s'],
            'type': ''
            })
    txt = correct_mistakes(txt, mistakes)
    print('speller:', txt)

    params = {'text': txt, 'language': language, 'ai': 0, 'key': CORR_KEY}
    response = requests.get(url="https://api.textgears.com/grammar", params=params)
    grammar_mistakes = response.json()['response']['errors']
    txt = correct_mistakes(txt, grammar_mistakes)
    print('grammar:', txt)

    res = remove_extra_spaces(txt)
    return res


def main():
    txt = """Глига. Ночь балота звок ет 2 акаржна. на 
поляко стоит вросшая в зеклю гоива 
срублесная избурака потояновшая 
т
вре пет сунеств енное а тико. запе ужое 
виесто стекла какой- 
по дрялью драхит 
т о м к колтки съетам е В избо. за 
нак рачанни. дерсвянти к Сталом. друг 
протв я руга. сидят два здор посктых 
борода Сибарсках крхика с густы т окланье т и 
ни к пьют оконко- горяжийй ча и 
о гракых панятит аллтики овых в хрухеч. о """

    txt1 = 'На этом примере можно увиддть\n что что исправленея работают харашо но не всеггда.'
    txt2 = 'Helo, howw are yuo?'
    print('yandex:', yandex_corr(txt1))
    print('before:', correction(txt1))


if __name__ == '__main__':
    main()