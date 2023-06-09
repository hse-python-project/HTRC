from correction import correction
from recognition import recognise, convert_to_text


def magic(filename='', text='', mode=1):
    if mode == 3:
        print(correction(text))
        return correction(text)
    pred = recognise(read_path=filename, draw_type='rect')
    txt = convert_to_text(pred)
    print(txt)
    return correction(txt) if mode == 1 else txt
