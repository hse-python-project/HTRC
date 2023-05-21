import easyocr
from correction import correction_with_punctuality,correct_text_final
from recognition import recognise

def magic_without_correction(filename):
    res = easy_ocr_recognition(filename)
    return res


def magic_with_correction(filename):
    txt = easy_ocr_recognition(filename)
    first_fix = correction_with_punctuality(txt)
    result = correct_text_final(first_fix, "ru")  # добавить исправление языка текста на английский при необходимости
    text = result[0]
    mistake_counter = result[1]
    return (text, mistake_counter)

def magic(filename, mode):
    res = recognise(read_path=filename, draw_type='rect')
    return ' '.join(box['text'] for box in res['predictions'])


def easy_ocr_recognition(file_path):
    reader = easyocr.Reader(["ru", "en"])
    result = reader.readtext(file_path, detail=0)

    string = ''
    for k in result:
        string += k + ' '

    return string
