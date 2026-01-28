import cv2 as cv
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
    return cv.cvtColor(image, cv.COLOR_BGR2GRAY)

def noise_removal(image):
    import numpy as np
    kernel = np.ones((1,1), np.uint8)
    image = cv.dilate(image,kernel, iterations=1)
    kernel = np.ones((1,1), np.uint8)
    image = cv.erode(image, kernel, iterations=1)
    image = cv.morphologyEx(image, cv.MORPH_CLOSE,kernel)
    image = cv.medianBlur(image,3)
    return (image)

def thin_font(image):
    import numpy as np
    image = cv.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv.erode(image,kernel,iterations=1)
    image = cv.bitwise_not(image)
    return (image)

def thicc_font(image):
    import numpy as np
    image = cv.bitwise_not(image)
    kernel = np.ones((2,2),np.uint8)
    image = cv.dilate(image,kernel,iterations=1)
    image = cv.bitwise_not(image)
    return (image)

import numpy as np

def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv.cvtColor(newImage, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(gray, (9, 9), 0)
    thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (30, 5))
    dilate = cv.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv.findContours(dilate, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv.contourArea, reverse = True)
    for c in contours:
        rect = cv.boundingRect(c)
        x,y,w,h = rect
        cv.rectangle(newImage,(x,y),(x+w,y+h),(0,255,0),2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print (len(contours))
    minAreaRect = cv.minAreaRect(largestContour)
    cv.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle
# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv.warpAffine(newImage, M, (w, h), flags=cv.INTER_CUBIC, borderMode=cv.BORDER_REPLICATE)
    return newImage

def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)

def remove_border(image):
    contours, heiarchy = cv.findContours(image,cv.RETR_EXTERNAL,cv.CHAIN_APPROX_SIMPLE)
    cnt_sorted = sorted(contours,key=lambda x:cv.contourArea(x))
    cnt = cnt_sorted[-1]
    x,y,w,h = cv.boundingRect(cnt)
    crop = img[y:y+h,x:x+w]
    return (crop)

img_file = "data/page_01.jpg"
img = cv.imread(img_file)

# cv.imshow("original image", img)
# cv.waitKey(10000)

#not necessery but need to know
invert_img = cv.bitwise_not(img)
cv.imwrite("temp/inverted01.jpg",invert_img)
display("temp/inverted01.jpg")

#convert image to greyscale
grey_imge = greyscale(img)
cv.imwrite("temp/grey_img01.jpg",grey_imge)
display("temp/grey_img01.jpg")

#black&white image
thresh, img_bw = cv.threshold(grey_imge, 211,230, cv.THRESH_BINARY)
cv.imwrite("temp/bw_img.jpg",img_bw)
display("temp/bw_img.jpg")

#noise removal
no_noise = noise_removal(img_bw)
cv.imwrite("temp/no__noise_img.jpg", no_noise)
display("temp/no__noise_img.jpg")

#thinig the font
erode_img = thin_font(no_noise)
cv.imwrite("temp/erode_img.jpg", erode_img)
display("temp/erode_img.jpg")

#thicken the font
dilate_img = thicc_font(no_noise)
cv.imwrite("temp/dilate_img.jpg", dilate_img)
display("temp/dilate_img.jpg")

#rotate and deskew (must remove border first)
rotate_img = cv.imread("data/page_01_rotated.JPG")
fixed = deskew(rotate_img)
cv.imwrite("temp/fixed.jpg",fixed)
display("data/page_01_rotated.JPG")
display("temp/fixed.jpg")

#remove border
no_border = remove_border(no_noise)
cv.imwrite("temp/no_border01.jpg",no_border)
display("temp/no_border01.jpg")

#add in missing border
color = [255,255,255]
top,bottom,left,right = [150]*4
img_with_border = cv.copyMakeBorder(no_border, top,bottom,left,right ,cv.BORDER_CONSTANT,value=color)
cv.imwrite("temp/img_with_border.jpg",img_with_border)
display("temp/img_with_border.jpg")