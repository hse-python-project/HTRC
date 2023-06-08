from correction import correction
from recognition import recognise, convert_to_text


def magic(filename, mode):
    pred = recognise(read_path=filename, draw_type='rect')
    txt = convert_to_text(pred)
    print(txt)
    return correction(txt) if mode == 1 else txt
