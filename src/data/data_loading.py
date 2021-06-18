from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
import random
import tqdm
import os

def get_paths(datapath, dirname):
    path = os.path.join(datapath, dirname)

    positives = os.path.join(path, 'positives')
    negatives = os.path.join(path, 'negatives')
    partials = os.path.join(path, 'partials')

    return [positives, negatives, partials]

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
        image = load_img(path, color_mode='rgb', target_size=(pixels, pixels))
        image = img_to_array(image) / 255
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