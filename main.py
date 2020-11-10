import cv2
import pytesseract
import numpy as np

def get_mask(img, lower=np.array([0, 0, 0]), upper=np.array([255, 255, 255])):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower, upper)

faceCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

img = cv2.imread('in.jpeg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.2,
    minNeighbors=5,
    minSize=(20, 20)
)
for (x, y, w, h) in faces:
    roi_color = img[y:y+h, x:x+w]

img = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)

img_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.png", img_gray)

img_gauss = cv2.GaussianBlur(img_gray, (3, 3), 0)
cv2.imwrite("gaussian_blur.png", img_gauss)

img_adapt = cv2.adaptiveThreshold(img_gauss, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 99, 4)
cv2.imwrite("adaptive_threshold.png", img_adapt)

print(pytesseract.image_to_string(img_adapt, lang='rus'))
