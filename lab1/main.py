import cv2

print("OpenCV version: " + cv2.__version__)

RgbImage = cv2.imread('./lab1data/ET_1.jpg')

print("Showing normal version image")
cv2.imshow('RgbImage', RgbImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

grayImage = cv2.cvtColor(RgbImage, cv2.COLOR_BGR2GRAY)
print("Showing gray version image")
cv2.imshow('grayImage', grayImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

smoothImage = cv2.blur(RgbImage, (5, 5))
print("Showing blur version image")
cv2.imshow('smoothImage', smoothImage)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Saving the blur version image")
directory = './output/smoothImage.jpg'
cv2.imwrite(directory, smoothImage)
