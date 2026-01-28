from PIL import Image

img_file = "data/page_01.jpg"
img = Image.open(img_file)
img.show()
img.save("temp/page_01.jpg")
img.rotate(90).show()
