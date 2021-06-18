import cv2
import numpy as np

def draw_bounding_box(image, box, colour=(255, 0, 0), thickness=2):
    x1 = box[0]
    x2 = x1 + box[2]
    y1 = box[1]
    y2 = y1 + box[3]
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, thickness=thickness)

def imshow(image):
    cv2.imshow('title', image[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()