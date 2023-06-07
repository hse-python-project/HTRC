from correction import correction_with_punctuation
from recognition import recognise, convert_to_text


def magic(filename, mode):
    pred = recognise(read_path=filename, draw_type='rect')
    txt = convert_to_text(pred)
    return correction_with_punctuation(txt) + "\nCORRECTED" if mode == 1 else txt
