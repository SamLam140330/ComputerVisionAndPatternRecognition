import cv2
from skimage import feature, exposure

print("Showing viola jones face detection")
face_cascade = cv2.CascadeClassifier('./lab3data/haarcascade_frontalface_default.xml')
img = cv2.imread('./lab3data/face/example.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

img = cv2.imread('./lab3data/person/crop001208.png')
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, hog_image = feature.hog(img, 8, (16, 16), (1, 1), visualize=True, channel_axis=2)
hog_image_rescaled = exposure.rescale_intensity(hog_image, (0, 10))

print("Showing histogram of oriented gradient")
cv2.imshow("HOG", hog_image_rescaled)
cv2.waitKey(0)
cv2.destroyAllWindows()

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
(rects, weights) = hog.detectMultiScale(gray_img, winStride=(4, 4), padding=(8, 8), scale=1.25,
                                        useMeanshiftGrouping=False)
for (x, y, w, h) in rects:
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

print("Showing person detection")
cv2.imshow("Person Detection", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
