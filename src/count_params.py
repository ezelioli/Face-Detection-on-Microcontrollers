from tensorflow.keras.models import load_model
from tensorflow.keras import backend
import numpy as np

def count_params(model):
	shapes_mem_count = 0
	for layer in model.layers:
		single_layer_mem = 1
		out_shape = layer.output_shape
		if type(out_shape) is list:
			out_shape = out_shape[0]
		for s in out_shape:
			if s is None:
				continue
			single_layer_mem *= s
		shapes_mem_count += single_layer_mem

	trainable_count = np.sum([backend.count_params(weights) for weights in model.trainable_weights])
	non_trainable_count = np.sum([backend.count_params(weights) for weights in model.non_trainable_weights])

	return (trainable_count, non_trainable_count, shapes_mem_count)


def report_params(params, size=4, batch_size=1):
	trainable_count, non_trainable_count, shapes_mem_count = params
	total_params = trainable_count + non_trainable_count
	print('TRAINABLE PARAMS     : %d' % (trainable_count))
	print('NON TRAINABLE PARAMS : %d' % (non_trainable_count))
	print('LAYER OUTPUT         : %d' % (shapes_mem_count))
	print('WEIGHTS FOOTPRINT    : %d KB' % (int(total_params * size / (1024 ** 1))))
	print('DATA FOOTPRINT       : %d KB' % (int(shapes_mem_count * batch_size * size / (1024 ** 1))))
	print('MEMORY FOOTPRINT     : %d KB' % (int((total_params + shapes_mem_count * batch_size) * size / (1024 ** 1))))


onet = load_model('./models/onet_relu.h5')
onet_params = count_params(onet)
report_params(onet_params)

rnet = load_model('./models/rnet_relu.h5')
rnet_params = count_params(rnet)
report_params(rnet_params)

pnet = load_model('./models/pnet.h5')
pnet_params = count_params(pnet)
report_params(pnet_params)
