import easyocr
import os


def magic(filename):
    txt = easy_ocr_recognition(filename)
    return txt

    # TODO: the ML stuff


#Egor   |  |  |
#       v  v  v

def easy_ocr_recognition(file_path):
  reader = easyocr.Reader(["ru", "en"])
  result = reader.readtext(file_path, detail=0)
  
  string = ''
  for k in result:
    string += k + ' '

  return string