import cv2
import pytesseract as pt
from matplotlib import pyplot as plt

def display(im_path):
    dpi = 80
    im_data = plt.imread(im_path)

    height, width  = im_data.shape[:2]
    
    # What size does the figure need to be in inches to fit the image?
    figsize = width / float(dpi), height / float(dpi)

    # Create a figure of the right size with one axes that takes up the full figure
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    # Hide spines, ticks, etc.
    ax.axis('off')

    # Display the image.
    ax.imshow(im_data, cmap='gray')

    plt.show()
def greyscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

img_file = "data/th_01.jpg"
img=cv2.imread(img_file)

grey_img = greyscale(img)
cv2.imwrite("temp/th_grey.jpg", grey_img)
display("temp/th_grey.jpg")

thresh, img_bw = cv2.threshold(grey_img, 127,255, cv2.THRESH_BINARY)
cv2.imwrite("temp/bw_img.jpg",img_bw)
display("temp/bw_img.jpg")

ocr_result = pt.image_to_string(img_bw,lang='tha')
print(ocr_result)

