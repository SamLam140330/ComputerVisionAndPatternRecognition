import cv2
import numpy as np

# Filtering
originalImg = cv2.imread('./lab2data/ET_1.jpg')

print("Showing low pass filter image")
kernel = np.ones((5, 5), np.float32) / 25
filterImg = cv2.filter2D(originalImg, -1, kernel)
cv2.imshow('SmoothImg', filterImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Showing medium filter image")
medianImg = cv2.medianBlur(originalImg, 5)
cv2.imshow('MedianImg', medianImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Showing edge detection image")
edges = cv2.Canny(originalImg, 100, 200)
cv2.imshow('EdgeImg', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Segmentation
grayImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2GRAY)

print("Showing threshold image")
_, threshImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)
cv2.imshow('BINARY ThreshImg', threshImg)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("Showing adaptive threshold image")
adaptiveThreshImg = cv2.adaptiveThreshold(grayImg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow('Adaptive Gaussian ThreshImg', adaptiveThreshImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y


def region_grow(img, all_seed, thresh, p=1):
    height, weight = img.shape
    seed_mark = np.zeros(img.shape)
    seed_list = []
    for seed in all_seed:
        seed_list.append(seed)
    label = 1
    connects = select_connects(p)
    while len(seed_list) > 0:
        current_point = seed_list.pop(0)
        seed_mark[current_point.x, current_point.y] = label
        for i in range(8):
            tmp_x = current_point.x + connects[i].x
            tmp_y = current_point.y + connects[i].y
            if tmp_x < 0 or tmp_y < 0 or tmp_x >= height or tmp_y >= weight:
                continue
            gray_diff = get_gray_diff(img, current_point, Point(tmp_x, tmp_y))
            if gray_diff < thresh and seed_mark[tmp_x, tmp_y] == 0:
                seed_mark[tmp_x, tmp_y] = label
                seed_list.append(Point(tmp_x, tmp_y))
    return seed_mark


def select_connects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
                    Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects


def get_gray_diff(img, current_point, tmp_point):
    return abs(int(img[current_point.x, current_point.y]) - int(img[tmp_point.x, tmp_point.y]))


print("Showing region growth image")
seeds = [Point(10, 10), Point(82, 150), Point(20, 300)]
binaryImg = region_grow(grayImg, seeds, 10)
cv2.imshow('regionGrowImg', binaryImg)
cv2.waitKey(0)
cv2.destroyAllWindows()


def number_of_clusters(k_value):
    stop_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
    ret, labels, clusters = cv2.kmeans(reshapedImage, k_value, None, stop_criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    clusters = np.uint8(clusters)
    intermediate_image = clusters[labels.flatten()]
    clustered_image = intermediate_image.reshape(RgbImg.shape)
    return clustered_image


print("Showing K-means image")
RgbImg = cv2.cvtColor(originalImg, cv2.COLOR_BGR2RGB)
reshapedImage = np.float32(RgbImg.reshape(-1, 3))
k = 4
clusteredImage = number_of_clusters(k)
cv2.imshow('Segmentation image with K=' + str(k), clusteredImage)
cv2.waitKey(0)
cv2.destroyAllWindows()
