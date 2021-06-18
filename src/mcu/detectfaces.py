from src.utils import imshow
import numpy as np
import random
import serial
import time
import cv2
import os


def draw_bounding_box(image, box, colour=(255, 0, 0), thickness=1):
    x1 = box[0]
    x2 = x1 + box[2]
    y1 = box[1]
    y2 = y1 + box[3]
    cv2.rectangle(image, (y1, x1), (y2, x2), colour, thickness=thickness)


def merge(coords, offset, w, h):
    assert len(coords) == len(offset) == 4, "Check coords and offset type"
    #offset[0:2] = (offset[0:2] / 128) * 50
    #offset[2:4] = offset[2:4] / 2
    offset[0:2] = (offset[0:2] / 128) * 15
    offset[2:4] = offset[2:4] / 4
    x1 = max(0, int(coords[0] + offset[0]))
    y1 = max(0, int(coords[1] + offset[1]))
    x2 = min(w, int(coords[2] + offset[3]))
    y2 = min(h, int(coords[3] + offset[2]))
    return np.array([x1, y1, x2, y2], dtype=np.int32)

def sendtomcu(image, ser):
    image = image.flatten()
    for byte in image:
        for char in str(byte):
            ser.write(char.encode())
            time.sleep(0.001)
        ser.write('\n'.encode())
        time.sleep(0.001)
    response = ser.read(size=20)
    print(response)
    response = response.decode().split(' ')
    detection = int(response[0])
    boxe = [int(x) for x in response[1:]]
    return detection, boxe


def detectfaces(impath):
    image = cv2.cvtColor(cv2.imread(impath), cv2.COLOR_BGR2RGB)
    assert image.shape == (128, 128, 3), "Image size must be 128x128x3"
    ser = serial.Serial(
        port='/dev/ttyACM0',
        baudrate=115200,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=6
    )
    assert ser.isOpen(), "Cannot open serial connection to ACM0"

    image_out = image.copy()
    #imshow(image)

    coords = np.array([[x, y, 44, 44] for x in range(0, 125, 42) for y in range(0, 125, 42)])
    #windows = np.array([cv2.resize(image[x:x+44, y:y+44, :], (24, 24), interpolation=cv2.INTER_AREA) for x in range(0, 125, 42) for y in range(0, 125, 42)])
    windows = np.array([cv2.resize(image[x:x+44, y:y+44, :], (48, 48), interpolation=cv2.INTER_AREA) for x in range(0, 125, 42) for y in range(0, 125, 42)])
    wins = []
    for window in windows:
        im_trans = np.transpose(window, (1, 0, 2))
        #im_norm = (im_trans - np.full((24, 24, 3), 128)) * 1
        im_norm = (im_trans - np.full((48, 48, 3), 128)) * 1
        im = im_norm.astype(np.int8)
        wins.append(im)

    windows = np.array(wins)

    detections = []
    boxes = []
    for win in windows:
        detection, boxe = sendtomcu(win, ser)
        detections.append(detection)
        boxes.append(boxe)
        time.sleep(2)

    #print(detections)
    #print(boxes)

    draw_boxes = []
    for score, box, coord in zip(detections, np.array(boxes), coords):
        if score > 0:
            newbox = merge(coord, box, 44, 44)
            draw_boxes.append(newbox)
            #boxes.append(coord)

    for box in draw_boxes:
        draw_bounding_box(image_out, box, (255, 255, 255))
    
    #imshow(image_out)
    cv2.imwrite(f'{random.randint(1, 9)}_{random.randint(10, 19)}.jpg', image_out)

    ser.close()

