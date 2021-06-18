from tensorflow.keras.models import load_model, model_from_json
import numpy as np
import cv2

def compute_scale_pyramid(m, min_layer, scale_factor: float = 0.709):
    scales = []
    factor_count = 0

    while min_layer >= 12:
        scales += [m * np.power(scale_factor, factor_count)]
        min_layer = min_layer * scale_factor
        factor_count += 1

    return scales

def scale_image(image, scale):
    """
    Scales the image to a given scale.
    :param image: image to be scaled
    :param scale: scale factor
    :return: scaled image
    """
    h, w, _ = image.shape

    w_scaled = int(np.ceil(w * scale))
    h_scaled = int(np.ceil(h * scale))

    image_resized = cv2.resize(image, (w_scaled, h_scaled), interpolation=cv2.INTER_AREA)

    image_normalized = (image_resized - 127.5) * 0.0078125

    return image_normalized

def generate_bounding_box(heat_map, regression_score, scale, t):
    """
    Bounding Boxes Generation.
    """
    stride = 2
    cellsize = 12

    heat_map = np.transpose(heat_map)
    dx1 = np.transpose(regression_score[:, :, 0])
    dy1 = np.transpose(regression_score[:, :, 1])
    dx2 = np.transpose(regression_score[:, :, 2])
    dy2 = np.transpose(regression_score[:, :, 3])

    y, x = np.where(heat_map >= t)

    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)

    score = heat_map[(y, x)]
    regression_score = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

    if regression_score.size == 0:
        regression_score = np.empty(shape=(0, 3))

    bb = np.transpose(np.vstack([y, x]))

    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), regression_score])

    return boundingbox, regression_score

def nms(boxes, threshold, method):
    """
    Non Maximum Suppression.

    :param boxes: numpy array with bounding boxes.
    :param threshold: selection threshold
    :param method: NMS method to apply. Available values ('Min', 'Union')
    """
    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    sorted_s = np.argsort(s)

    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0
    while sorted_s.size > 0:
        i = sorted_s[-1]
        pick[counter] = i
        counter += 1
        idx = sorted_s[0:-1]

        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h

        if method == 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        sorted_s = sorted_s[np.where(o <= threshold)]

    pick = pick[0:counter]

    return pick

def rerec(bbox):
    # convert bbox to square
    height = bbox[:, 3] - bbox[:, 1]
    width = bbox[:, 2] - bbox[:, 0]
    max_side_length = np.maximum(width, height)
    bbox[:, 0] = bbox[:, 0] + width * 0.5 - max_side_length * 0.5
    bbox[:, 1] = bbox[:, 1] + height * 0.5 - max_side_length * 0.5
    bbox[:, 2:4] = bbox[:, 0:2] + np.transpose(np.tile(max_side_length, (2, 1)))
    return bbox

def bbreg(boundingbox, reg):
    # calibrate bounding boxes
    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox

def extract_image_area(image, box, size):
    tempimg = np.zeros((24, 24, 3))

    h, w, _ = image.shape

    b0, b1, b2, b3 = box.copy().astype(np.int32)

    tmpw, tmph = b2 - b0, b3 - b1

    tmp = np.zeros((tmph, tmpw, 3))

    x1 = max(0, b0)
    y1 = max(0, b1)
    x2 = min(w, b2)
    y2 = min(h, b3)

    dx = 0
    if b0 < 0:
        dx = - b0
    dy = 0
    if b1 < 0:
        dy = - b1
    edx = tmpw
    if b2 > w:
        edx = tmpw - (b2 - w)
    edy = tmph
    if b3 > h:
        edy = tmph - (b3 - h)

    tmp[dy:edy, dx:edx, :] = image[y1:y2, x1:x2, :]

    if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
        tempimg = cv2.resize(tmp, (size, size), interpolation=cv2.INTER_AREA)

    else:
        return np.empty(shape=(0,))
    
    return tempimg

def stage_one(image, scales, pnet, threshold: float = 0.6):
    """
    First stage of the MTCNN.
    """
    total_boxes = np.empty((0, 9))

    for scale in scales:
        scaled_image = scale_image(image, scale)

        img_x = np.expand_dims(scaled_image, 0)
        img_y = np.transpose(img_x, (0, 2, 1, 3))

        out = pnet.predict(img_y)

        out_regressor = np.transpose(out[0], (0, 2, 1, 3))
        out_classifier = np.transpose(out[1], (0, 2, 1, 3))

        boxes, _ = generate_bounding_box(out_classifier[0, :, :, 1].copy(),
                                            out_regressor[0, :, :, :].copy(), scale, threshold)

        pick = nms(boxes.copy(), 0.5, 'Union')
        if boxes.size > 0 and pick.size > 0:
            boxes = boxes[pick, :]
            total_boxes = np.append(total_boxes, boxes, axis=0)

    numboxes = total_boxes.shape[0]

    if numboxes > 0:
        pick = nms(total_boxes.copy(), 0.7, 'Union')
        total_boxes = total_boxes[pick, :]

        regw = total_boxes[:, 2] - total_boxes[:, 0]
        regh = total_boxes[:, 3] - total_boxes[:, 1]

        qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
        qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
        qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
        qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh

        total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
        total_boxes = rerec(total_boxes.copy())

        total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)

    return total_boxes

def stage_two(image, total_boxes, rnet, threshold: float = 0.7):
    """
    Second stage of the MTCNN.
    """

    num_boxes = total_boxes.shape[0]
    if num_boxes == 0:
        return total_boxes

    tempimg = np.zeros(shape=(24, 24, 3, num_boxes))

    for k in range(0, num_boxes):
        box = total_boxes[k, 0:4]
        tempimg[:, :, :, k] = extract_image_area(image, box, 24)

    tempimg = (tempimg - 127.5) * 0.0078125
    tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

    out = rnet.predict(tempimg1)

    out_regressor = np.transpose(out[0])
    out_classifier = np.transpose(out[1])

    score = out_classifier[1, :]

    ipass = np.where(score > threshold)

    total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

    mv = out_regressor[:, ipass[0]]

    if total_boxes.shape[0] > 0:
        pick = nms(total_boxes, 0.7, 'Union')
        total_boxes = total_boxes[pick, :]
        total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
        total_boxes = rerec(total_boxes.copy())

    return total_boxes

def stage_three(image, total_boxes, onet, threshold: float = 0.7):
    """
    Third stage of the MTCNN.
    """
    num_boxes = total_boxes.shape[0]
    if num_boxes == 0:
        print('no boxes to onet')
        return np.empty(shape=(0,))

    total_boxes = np.fix(total_boxes).astype(np.int32)

    tempimg = np.zeros((48, 48, 3, num_boxes))

    for k in range(0, num_boxes):
        box = total_boxes[k, 0:4]
        tempimg[:, :, :, k] = extract_image_area(image, box, 48)

    tempimg = (tempimg - 127.5) * 0.0078125
    tempimg1 = np.transpose(tempimg, (3, 1, 0, 2))

    out = onet.predict(tempimg1)
    out0 = np.transpose(out[0])
    out1 = np.transpose(out[1])
    #out1 = np.transpose(out[2])

    score = out1[1, :]

    ipass = np.where(score > threshold)

    total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])

    mv = out0[:, ipass[0]]

    if total_boxes.shape[0] > 0:
        total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
        pick = nms(total_boxes.copy(), 0.7, 'Min')
        total_boxes = total_boxes[pick, :]

    return total_boxes

def detect_faces(image, min_face_size: int = 20):
        """
        Detects bounding boxes from the specified image.
        :param image: image to process
        :return: list containing all the bounding boxes detected with their keypoints.
        """
        if image is None or not hasattr(image, "shape"):
            raise Exception("Image not valid.")

        pnet = load_model('./models/pnet.h5', compile=False)
        rnet = load_model('./models/rnet.h5', compile=False)
        onet = load_model('./models/onet.h5', compile=False)

        h, w, _ = image.shape

        m = 12 / min_face_size
        min_layer = np.amin([h, w]) * m

        scales = compute_scale_pyramid(m, min_layer)

        total_boxes_pnet = stage_one(image, scales, pnet)

        total_boxes_rnet = stage_two(image, total_boxes_pnet, rnet)

        total_boxes_onet = stage_three(image, total_boxes_rnet, onet)

        bounding_boxes_pnet = []

        for bounding_box in total_boxes_pnet:
            x = max(0, int(bounding_box[0]))
            y = max(0, int(bounding_box[1]))
            w = int(bounding_box[2] - x)
            h = int(bounding_box[3] - y)
            bounding_boxes_pnet.append({
                'box': [x, y, w, h],
                'confidence': bounding_box[-1]
            })
        
        bounding_boxes_rnet = []

        for bounding_box in total_boxes_rnet:
            x = max(0, int(bounding_box[0]))
            y = max(0, int(bounding_box[1]))
            w = int(bounding_box[2] - x)
            h = int(bounding_box[3] - y)
            bounding_boxes_rnet.append({
                'box': [x, y, w, h],
                'confidence': bounding_box[-1]
            })

        bounding_boxes_onet = []
        print(total_boxes_onet)
        for bounding_box in total_boxes_onet:
            x = max(0, int(bounding_box[0]))
            y = max(0, int(bounding_box[1]))
            w = int(bounding_box[2] - x)
            h = int(bounding_box[3] - y)
            bounding_boxes_onet.append({
                'box': [x, y, w, h],
                'confidence': bounding_box[-1]
            })

        return bounding_boxes_pnet, bounding_boxes_rnet, bounding_boxes_onet
