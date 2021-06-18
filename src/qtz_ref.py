from tensorflow.keras.models import load_model
from src.mcu.gen_header import hex_to_c_array
from src.data.data_loading import load_data
import tensorflow as tf
import numpy as np
import argparse
import random
import tqdm
import cv2
import os


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
    print(input_scale)
    print(input_zero_point)
    print(input_details[0]['index'])


def main(args):
    modelpath = args.modelpath
    modelname = args.modelname
    datapath = args.datapath
    pixels = args.pixels
    size = args.size

    model = load_model(modelpath)
    model.summary()
    val_data, val_categories, val_bboxes = load_data(datapath, train=False, size=size, pixels=pixels)

    def representative_dataset():
        for i in range(len(val_data)):
         yield [np.expand_dims(val_data[i], axis=0)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8  # or tf.uint8
    converter.inference_output_type = tf.int8  # or tf.uint8
    tflite_quant_model = converter.convert()

    open(f'./models/{modelname}.tflite', 'wb').write(tflite_quant_model)

    interpreter = tf.lite.Interpreter(f'./models/{modelname}.tflite')
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print_io_details(interpreter)

    with open(f'./models/{modelname}.h', 'w') as file:
        file.write(hex_to_c_array(tflite_quant_model, modelname))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='modelpath', type=str, help="Path of the model to quantize")
    parser.add_argument(dest='datapath', type=str, help="Path of the data directory")
    parser.add_argument('-s', '--size', dest='size', type=int, default=500)
    parser.add_argument('-p', '--pixels', dest='pixels', type=int, default=24)
    parser.add_argument('-n', '--name', dest='modelname', type=str, default='onet')

    args = parser.parse_args()

    main(args)



