from correction import correct
from recognition import recognise, convert_to_text


def magic(filename='', text='', mode=1):
    if mode == 3:
        return correct(text)
    pred = recognise(read_path=filename, draw_type='rect')
    txt = convert_to_text(pred)
    return correct(txt) if mode != 2 else txt
