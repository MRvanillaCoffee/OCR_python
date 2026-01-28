import pytesseract as pyt
from PIL import Image

#simple ocr with only no_noise img
img_file = "data/page_01.jpg"
no_noise = "temp/dilate_img.jpg"
img = Image.open(img_file)
nn = Image.open(no_noise)
ocr_result = pyt.image_to_string(nn)
print(ocr_result)

