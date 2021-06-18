from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from src.mcu.gen_header import hex_to_c_array
from src.data.data_loading import get_paths
import tensorflow as tf
import numpy as np
import random
import tqdm
import cv2
import os

MODELS = {
    'pnet' : './models/pnet_relu.h5',
    'rnet' : './models/rnet_relu.h5',
    'onet' : './models/onet_relu.h5',
    'pnet_tflite' : './models/pnet.tflite',
    'rnet_tflite' : './models/rnet.tflite',
    'onet_tflite' : './models/onet.tflite',
    'pnet_h' : './models/pnet.h',
    'rnet_h' : './models/rnet.h',
    'onet_h' : './models/onet.h'
}

VAL_BATCH_SIZE = 4

def load_models():
    pnet = load_model(MODELS['pnet'])
    rnet = load_model(MODELS['rnet'])
    onet = load_model(MODELS['onet'])
    return pnet, rnet, onet

def evaluate(model_path, test_set=None, test_labels=None, score=None):
    tflite_interpreter = tf.lite.Interpreter(model_path=model_path)
    tflite_interpreter.allocate_tensors()
    input_details = tflite_interpreter.get_input_details()
    output_details = tflite_interpreter.get_output_details()

    print("== Input details ==")
    print("name:", input_details[0]['name'])
    print("shape:", input_details[0]['shape'])
    print("type:", input_details[0]['dtype'])

    print("\n== Output details ==")
    print("name:", output_details[0]['name'])
    print("shape:", output_details[0]['shape'])
    print("type:", output_details[0]['dtype'])

    predictions = np.zeros((len(test_set),), dtype=int)
    input_scale, input_zero_point = input_details[0]["quantization"]
    for i in range(len(test_set)):
        val_batch = test_set[i]
        val_batch = val_batch / input_scale + input_zero_point
        val_batch = np.expand_dims(val_batch, axis=0).astype(input_details[0]["dtype"])
        tflite_interpreter.set_tensor(input_details[0]['index'], val_batch)
        tflite_interpreter.allocate_tensors()
        tflite_interpreter.invoke()

        tflite_model_predictions = tflite_interpreter.get_tensor(output_details[0]['index'])
        #print("Prediction results shape:", tflite_model_predictions.shape)
        output = tflite_interpreter.get_tensor(output_details[0]['index'])
        predictions[i] = output.argmax()

    sum = 0
    for i in range(len(predictions)):
        if (predictions[i] == test_labels[i]):
            sum = sum + 1
    accuracy_score = sum / 100
    print("Accuracy of quantized to int8 model is {}%".format(accuracy_score*100))
    print("Compared to float32 accuracy of {}%".format(score[1]*100))
    print("We have a change of {}%".format((accuracy_score-score[1])*100))

def compile_evaluate(model, val_data, val_categories, val_bboxes, outputs, val_batch_size=4):
    classification, regression = outputs

    losses = {
        classification : tf.keras.losses.BinaryCrossentropy(),
        regression : tf.keras.losses.MeanSquaredError()
    }
    loss_weights = {
        classification : 1.0,
        regression : 0.5
    }

    model.compile(
        loss=losses,
        loss_weights=loss_weights,
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
        metrics=['accuracy', 'mse']
    )

    if val_data.shape[1] == 64:
        return None
    score = model.evaluate(
        x=val_data,
        y={
            classification : val_categories,
            regression : val_bboxes
            },
        batch_size=val_batch_size
    )
    return score

def convert_model(model, representative_dataset):
    # Generate converter for model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Enforce full-int8 quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8

    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = representative_dataset

    # Convert model
    model_tflite = converter.convert()

    return model_tflite

def load_data(datapath, train=True, size=-1, pixels=12):
    if train:
        target = 'train'
    else:
        target = 'val'

    print(f'INFO: Loading {target} images ...')
    
    paths = get_paths(datapath, target)
    root = os.path.join(datapath, f'{target}')

    samples = []
    for path in paths:
        labels_path = os.path.join(path, 'labels.txt')
        with open(labels_path, 'r') as file:
            lines = file.readlines()
            samples += [line[:-1] for line in lines[:]]
    
    random.seed(12)
    random.shuffle(samples)

    data = []
    categories = []
    bboxes = []
    for sample in tqdm.tqdm(samples):
        if size > 0 and len(data) >= size:
            break
        cat = [0, 0]
        tokens = sample.split(' ')
        rel_path, cat[0], cat[1] = tokens[:3]
        cat = [int(x) for x in cat]
        path = os.path.join(root, rel_path)
        image = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        image = (image - 127.5) * 0.0078125
        data.append(image)
        categories.append(cat)
        if cat[1] == 0:
            bbx = tokens[3:]
            bbx = [float(x) for x in bbx]
            bboxes.append((bbx[0], bbx[1], bbx[2], bbx[3]))
        else:
            bboxes.append((0.0,0.0,0.0,0.0))
    
    data = np.array(data, dtype=np.float32)
    categories = np.array(categories, dtype=np.float32)
    bboxes = np.array(bboxes, dtype=np.float32)

    return data, categories, bboxes

def load_val_data(val_samples=500):
    data = []
    for dim in [12, 24, 48]:
        val_data, val_categories, val_bboxes = load_data(f'./data/data_{dim}', train=False, size=val_samples, pixels=dim)

        data.append((val_data, val_categories, val_bboxes))
    return data

    

models = load_models()
data = load_val_data(500)
scores = []

pixels = [12, 24, 48]
outputs = [('softmax', 'conv2d_3'),
           ('softmax_1', 'dense_2'),
           ('softmax', 'dense_2')]

tflite_models = []

val_set = data[0]
val_data = val_set[0]
image = val_data[0]

for model, val_set, pixel, output in zip(models, data, pixels, outputs):
    val_data, val_categories, val_bboxes = val_set
    score = compile_evaluate(model, val_data, val_categories, val_bboxes, output)
    scores.append(score)
    def representative_dataset():
        for i in range(500):
            yield([val_data[i].reshape(1, pixel, pixel, 3)])
    tflite_model = convert_model(model, representative_dataset)
    tflite_models.append(tflite_model)

for model, name, cname in zip(tflite_models, ['pnet_tflite', 'rnet_tflite', 'onet_tflite'], ['pnet', 'rnet', 'onet']):
    # Save TFLite model
    open(MODELS[name], 'wb').write(model)

    # Write TFLite model to a C source (or header) file
    with open(MODELS[f'{cname}_h'], 'w') as file:
        file.write(hex_to_c_array(model, cname))

for score in scores:
    print(score)

# # Evaluate converted model
# evaluate(MODELS['pnet_tflite'], val_data[:500], )



# losses = {
#     # 'FACE_CLASSIFIER' : tf.keras.losses.BinaryCrossentropy(),
#     # 'BB_REGRESSION' : tf.keras.losses.MeanSquaredError()
#     'softmax' : tf.keras.losses.BinaryCrossentropy(),
#     'conv2d_3' : tf.keras.losses.MeanSquaredError()
# }
# loss_weights = {
#     # 'FACE_CLASSIFIER' : 1.0,
#     # 'BB_REGRESSION' : 0.5
#     'softmax' : 1.0,
#     'conv2d_3' : 0.5
# }

# pnet.compile(
#     loss=losses,
#     loss_weights=loss_weights,
#     optimizer=tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
#     metrics=['accuracy', 'mse']
# )

# score = pnet.evaluate(
#     x=val_data[:500],
#     # y={
#     #     'FACE_CLASSIFIER' : val_categories[:500],
#     #     'BB_REGRESSION' : val_bboxes[:500]
#     #     },
#     y={
#         'softmax' : val_categories,
#         'conv2d_3' : val_bboxes
#         },
#     batch_size=VAL_BATCH_SIZE
# )

# # Generate converter for model
# converter = tf.lite.TFLiteConverter.from_keras_model(pnet)

# # Set the optimization flag.
# converter.optimizations = [tf.lite.Optimize.DEFAULT]

# # Enforce full-int8 quantization
# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# converter.inference_input_type = tf.int8  # or tf.uint8
# converter.inference_output_type = tf.int8  # or tf.uint8

# # Provide a representative dataset to ensure we quantize correctly.
# converter.representative_dataset = representative_dataset

# # Convert model
# pnet_tflite = converter.convert()