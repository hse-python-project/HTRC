from correction import correction, yandex_corr
from recognition import recognise, convert_to_text


def magic(filename='', text='', mode=1):
    if mode == 3:
        print(correction(text))
        return yandex_corr(text)
    pred = recognise(read_path=filename, draw_type='rect')
    txt = convert_to_text(pred)
    print(txt)
    return yandex_corr(txt) if mode != 2 else txt
