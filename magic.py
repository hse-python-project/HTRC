def magic(filename):
    with open(filename, 'rb') as original:
        img = original.read()

    # TODO: the ML stuff

    return img

#Egor   |  |  |
#       v  v  v

import easyocr

def easy_ocr_recognition(file_path):
  reader = easyocr.Reader(["ru", "en"])
  result = reader.readtext(file_path, detail = 0)
  
  string = ''
  for k in result:
    string += k + ' '

  return string
