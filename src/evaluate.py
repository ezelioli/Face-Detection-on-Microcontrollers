from tensorflow.keras.models import load_model
from src.data.data_loading import load_data
from src.utils import imshow
import tensorflow as tf
import numpy as np
import random
import tqdm
import cv2
import os

def eval(modelpath, data, batch_size=1):
    model = load_model(modelpath)
    model.summary()

    val_data, val_categories, val_bboxes = data

    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in tqdm.tqdm(range(0,len(val_data), batch_size)):
        if batch_size == 1:
            batch = np.expand_dims(val_data[i], axis=0)
        else:
            batch = val_data[i:i+batch_size]
        bboxes, scores = model.predict(batch)
        for j in range(batch_size):
            gt = val_categories[i+j][0]
            if scores[j][0] > 0.7 and gt == 0:
                fp += 1
            if scores[j][0] <= 0.7 and gt == 1:
                fn += 1
            if scores[j][0] > 0.7 and gt == 1:
                tp += 1
            if scores[j][0] <= 0.7 and gt == 0:
                tn += 1
    return (fp, fn, tp, tn)

def eval_pnet():
	pnet = load_model('./models/pnet.h5')
	
	val_data, val_categories, val_bboxes = load_data('./data/data_64', train=False, size=500, pixels=48)
	
	for i in tqdm.tqdm(range(len(val_data))):
		batch = np.expand_dims(val_data[i], 0)
		bboxes, scores = pnet.predict(batch)
		print(bboxes.shape)
		print(scores.shape)
		exit()

def eval_framework():
	pnet = load_model('./models/pnet.h5')
	rnet = load_model('./models/rnet.h5')
	onet = load_model('./models/onet.h5')

	val_data, val_categories, val_bboxes = load_data('./data/data_64', train=False, size=512, pixels=48)

	errs = 0
	for i in tqdm.tqdm(range(0,len(val_data))):
		batch = np.expand_dims(val_data[i], axis=0)
		bboxes, scores = model.predict(batch)
		if abs(val_categories[0][0] - scores[0][0]) > 0.5:
			errs += 1
	print('Overall classification accuracy: %.2f' % (1 - errs / 500))

def print_io_details(interpreter):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("- bb regression")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])
    print("- classification")
    print("name:", output_details[1]['name'])
    print("shape:", output_details[1]['shape'])
    print("type:", output_details[1]['dtype'])

    input_scale, input_zero_point = input_details[0]["quantization"]

def eval_tflite(modelpath, data, batch_size=1):
    val_data, val_categories, val_bboxes = data
    interpreter = tf.lite.Interpreter(modelpath)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    #input_scale, input_zero_point = input_details[0]["quantization"]
    #input_dtype = input_details[0]["dtype"]
    print_io_details(interpreter)
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    for i in tqdm.tqdm(range(len(val_data))):
        image = np.expand_dims(val_data[i], axis=0)
        interpreter.set_tensor(input_details[0]['index'], image)
        interpreter.allocate_tensors()
        interpreter.invoke()
        bboxes = interpreter.get_tensor(output_details[0]['index'])
        score = interpreter.get_tensor(output_details[1]['index'])
        #print(score[0], val_categories[i][1])
        gt = val_categories[i][1]
        if score[0, 1] > 0 and gt == 0:
            fp += 1
        if score[0, 1] <= 0 and gt == 1:
            fn += 1
        if score[0, 1] > 0 and gt == 1:
            tp += 1
        if score[0, 1] <= 0 and gt == 0:
            tn += 1

    return (fp, fn, tp, tn)

def preprocess(data, scale, shift, dtype):
    images = []
    for i in range(len(data)):
        image = data[i]
        image_transposed = np.transpose(image, (1, 0, 2))
        image_normalized = (image_transposed - shift) * scale
        images.append(image_normalized.astype(dtype))
    return np.array(images, dtype=dtype)

def report(r, tot):
    fp, fn, tp, tn = r
    print('\tFalse positives   : %.4f' % (fp / tot))
    print('\tFalse negatives   : %.4f' % (fn / tot))
    print('\tTrue  positives   : %.4f' % (tp / tot))
    print('\tTrue  negatives   : %.4f' % (tn / tot))
    print('\tOverall accuracy  : %.4f' % ((tn + tp) / tot))

n_samples = 1024 * 5

modelpath = './models/onet.h5'
datapath = './data/data_48'
val_data, val_categories, val_bboxes = load_data(datapath, train=False, size=n_samples)
data = preprocess(val_data, 0.0078125, 127.5, np.float32)
r_onet = eval(modelpath, (data, val_categories, val_bboxes), batch_size=16)

modelpath = './models/rnet.h5'
datapath = './data/data_24'
val_data, val_categories, val_bboxes = load_data(datapath, train=False, size=n_samples)
data = preprocess(val_data, 0.0078125, 127.5, np.float32)
r_rnet = eval(modelpath, (data, val_categories, val_bboxes), batch_size=16)

modelpath = './models/onet_relu.h5'
datapath = './data/data_48'
val_data, val_categories, val_bboxes = load_data(datapath, train=False, size=n_samples)
data = preprocess(val_data, 0.0078125, 127.5, np.float32)
r_onet_relu = eval(modelpath, (data, val_categories, val_bboxes), batch_size=16)

modelpath = './models/rnet_relu.h5'
datapath = './data/data_24'
val_data, val_categories, val_bboxes = load_data(datapath, train=False, size=n_samples)
data = preprocess(val_data, 0.0078125, 127.5, np.float32)
r_rnet_relu = eval(modelpath, (data, val_categories, val_bboxes), batch_size=16)

modelpath = './tmp/models/onet.tflite'
datapath = './data/data_48'
val_data, val_categories, val_bboxes = load_data(datapath, train=False, size=n_samples)
data = preprocess(val_data, 1, 127.5, np.int8)
r_onet_lite = eval_tflite(modelpath, (data, val_categories, val_bboxes), batch_size=1)

modelpath = './tmp/models/rnet.tflite'
datapath = './tmp/data/data_24'
val_data, val_categories, val_bboxes = load_data(datapath, train=False, size=n_samples)
data = preprocess(val_data, 1, 127.5, np.int8)
r_rnet_lite = eval_tflite(modelpath, (data, val_categories, val_bboxes), batch_size=1)

print("Evaluation report:")
print('\nONET:')
report(r_onet, n_samples)
print('\nRNET')
report(r_rnet, n_samples)
print('\nONET RELU:')
report(r_onet_relu, n_samples)
print('\nRNET RELU:')
report(r_rnet_relu, n_samples)
print('\nONET TFLITE:')
report(r_onet_lite, n_samples)
print('\nRNET TFLITE:')
report(r_rnet_lite, n_samples)