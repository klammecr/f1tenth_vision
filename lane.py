import cv2
import numpy as np


def segment_lane(img, h_min = 21, h_max = 31, v_min = 89, v_max = 255, s_min = 56, s_max = 157):
    hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_img, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
    mask_close = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (5,5))
    # cv2.imshow("Masked Image", cv2.bitwise_and(img, img, mask=mask_close))
    # cv2.waitKey()
    contour_out = cv2.findContours(mask_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img_out = cv2.drawContours(img, contour_out[0], -1, (0,255,0), 3)
    cv2.imwrite("output/contour_img.png", img_out)
    # cv2.imshow("Contours", img_out)
    # cv2.waitKey()

if __name__ == "__main__":
    img = cv2.imread("resource/lane.png")
    segment_lane(img)