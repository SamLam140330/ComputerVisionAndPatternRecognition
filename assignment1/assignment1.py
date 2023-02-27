import cv2
import numpy as np


def flood_fill(image):
    copied_image = image.copy()
    height, width = image.shape[:2]
    mask = np.zeros((height + 2, width + 2), np.uint8)
    top_left_seed_point = (10, 10)
    fill_color = (0, 0, 0)
    lower_diff = (120, 120, 120)
    upper_diff = (150, 150, 150)
    flag = cv2.FLOODFILL_FIXED_RANGE
    cv2.floodFill(copied_image, mask, top_left_seed_point, fill_color, lower_diff, upper_diff, flag)
    return copied_image


def main():
    img = cv2.imread('./data/ori_img.jpg')
    output = flood_fill(img)
    print("Showing removed background image")
    cv2.imshow('Removed Background Image', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
