def magic(filename):
    with open(filename, 'rb') as original:
        img = original.read()

    # TODO: the ML stuff

    return img


import easyocr

def easy_ocr_recognition(file_path):
  reader = easyocr.Reader(["ru", "en"])
  result = reader.readtext(file_path, detail = 0)
  
  string = ''
  for k in result:
    string += k + ' '

  return string


def main():
  file_path = input('Введите путь до файла ')
  print(easy_ocr_recognition(file_path = file_path))

if __name__ == "__main__":
  main()
