import numpy.random as random
import numpy as np
import argparse
import tqdm
import cv2
import os


def parse_labels(filepath, size=-1):
    with open(filepath, 'r') as file:
        lines = file.readlines()
    labels = []
    i = 0
    while i < len(lines):
        assert lines[i].endswith('jpg\n'), f'ERROR: wrong format for labels file @ line {i}'
        if size > 0 and len(labels) >= size:
            break
        sample = {}
        sample['path'] = lines[i][:-1]
        i += 1
        n_faces = int(lines[i][:-1])
        if n_faces == 0:
            i += 2
            continue
        sample['n_faces'] = n_faces
        i += 1
        bboxes = []
        for j in range(i, i + n_faces):
            bbox = [int(val) for val in lines[j].split(' ')[:4]]
            bboxes.append(bbox)
        i += n_faces
        sample['bbox'] = bboxes
        labels.append(sample)
    
    return labels

def intersection_over_union(box, bboxes):
    box_area = box[2] * box[3]
    gt_area = bboxes[:, 2]* bboxes[:, 3]
    
    if bboxes.size == 0 or min(gt_area) <= 0:
        return None, None

    x1 = np.maximum(box[0], bboxes[:, 0])
    y1 = np.maximum(box[1], bboxes[:, 1])
    x2 = np.minimum(box[0] + box[2], bboxes[:, 0] + bboxes[:, 2])
    y2 = np.minimum(box[1] + box[3], bboxes[:, 1] + bboxes[:, 3])

    w = np.maximum(0, x2 - x1 + 1)
    h = np.maximum(0, y2 - y1 + 1)

    intersection = w * h
    union = box_area + gt_area - intersection

    iou = intersection / union
    overlap = intersection / gt_area
    
    return iou, overlap

def generate_negatives(image, bboxes, labels_path, path, imname, pixels, n_max=15, verbose=False):
    im_h, im_w, _ = image.shape
    n_neg = 0
    errs = 0
    while n_neg < n_max and errs < 50:
        size = random.randint(40, high=min(im_w, im_h)/2)
        x = random.randint(0, im_w - size)
        y = random.randint(0, im_h - size)
        w = size
        h = size

        if x is None or y is None:
            errs += 1
            if verbose:
                print('WARNING: x or y is None')
            continue

        box = np.array([x, y, w, h])

        iou, overlap = intersection_over_union(box, bboxes)
        iou, overlap = np.max(iou), np.max(overlap)

        if iou is None or overlap is None:
            errs += 1
            if verbose:
                print('WARNING: iou or overlap is None')
            continue

        if iou < 0.3 and overlap < 0.65:
            img_cropped = image[y : y + h, x : x + w, :]
            img_resized = cv2.resize(img_cropped, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
            dest_path = os.path.join(path, f'{imname}_{n_neg}.jpg')
            labels_path.write(f'negatives/{imname}_{n_neg}.jpg' + ' 0 1\n')
            cv2.imwrite(dest_path, img_resized)
            n_neg += 1
    
    return n_neg

def generate_positives(image, bboxes, labels_paths, paths, imname, pixels, min_face_size=20, verbose=False):
    n_bbox = -1
    tot_pos = 0
    tot_part = 0
    im_h, im_w, _ = image.shape
    positives, partials = paths
    labels_path_pos, labels_path_part = labels_paths
    for bbox in bboxes:
        x1, y1, w, h = bbox

        x2 = x1 + w
        y2 = y1 + h

        n_bbox += 1

        if max(im_w,im_h) < 40:
            if verbose:
                print(f'WARNING: skipping small image ({imname})')
            continue
            
        n_part = 0
        n_pos = 0
        errs = 0
        while (n_part < 1 or n_pos < 1) and errs < 50:
            size_min = int(min(w,h) * 0.8)
            size_max = np.ceil(1.25 * max(w,h))
            if size_min < min_face_size or size_max < min_face_size:
                if verbose:
                    print(f'WARNING: skipping too small face ({imname}_{n_bbox})')
                break
            size = random.randint(size_min, high=size_max)
            delta_x = random.randint(-w * 0.2, w * 0.2)
            delta_y = random.randint(-h * 0.2, h * 0.2)

            nx1 = int(max(x1 + w / 2 + delta_x - size / 2, 0))
            ny1 = int(max(y1 + h / 2 + delta_y - size / 2, 0))
            nx2 = nx1 + size
            ny2 = ny1 + size

            if nx2 > im_w or ny2 > im_h:
                if verbose:
                    print('WARNING: skipping out-of-boundaries image window')
                errs += 1
                if errs >= 50:
                    break
                continue
            box = np.array([nx1, ny1, size, size])

            offset_x1 = (x1 - nx1) / float(size)
            offset_y1 = (y1 - ny1) / float(size)
            offset_x2 = (x2 - nx2) / float(size)
            offset_y2 = (y2 - ny2) / float(size)

            img_cropped = image[ny1: ny2, nx1: nx2, :]
            img_resized = cv2.resize(img_cropped, (pixels, pixels), interpolation=cv2.INTER_LINEAR)
                
            iou, overlap = intersection_over_union(box, np.array([bbox]))
            
            if (iou >= 0.65 or overlap > 0.65) and n_pos < 1:
                dest_path = os.path.join(positives, f'{imname}_{n_bbox}_{n_pos}.jpg')
                labels_path_pos.write(f'positives/{imname}_{n_bbox}_{n_pos}.jpg' +
                    ' 1 0 %.2f %.2f %.2f %.2f\n' % ( offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(dest_path, img_resized)
                n_pos += 1
            elif iou >= 0.4 and n_part < 2:
                dest_path = os.path.join(partials, f'{imname}_{n_bbox}_{n_part}.jpg')
                labels_path_part.write(f'partials/{imname}_{n_bbox}_{n_part}.jpg' +
                     ' 1 0 %.2f %.2f %.2f %.2f\n' % (offset_x1, offset_y1, offset_x2, offset_y2))
                cv2.imwrite(dest_path, img_resized)
                n_part += 1
            else:
                errs += 1
        tot_pos += n_pos
        tot_part += n_part
    
    return tot_pos, tot_part

def generate_data(wider_path, root, size, train=True, n_images=-1, verbose=False):
    """
    Generates training data for pnet, rnet and onet
    :param wider_path: path of the root directory of wider face dataset on filesystem
    :param root      : path to data directory, where images will be generated
    :param size      : size of the images to generate [12, 24, 48, 64]
    :param train     : if True generates training data, else generates validation data
    """
    if train:
        dirname = 'train'
    else:
        dirname = 'val'
    
    outdir    = os.path.join(root, f'data_{size}')  # ./data/data_[12, 24, 48]
    path      = os.path.join(outdir, f'{dirname}')    # ./data/data_[12, 24, 48]/[train, val]
    positives = os.path.join(path, 'positives')  # ./data/data_[12, 24, 48]/[train, val]/positives
    negatives = os.path.join(path, 'negatives')  # ./data/data_[12, 24, 48]/[train, val]/negatives
    partials  = os.path.join(path, 'partials')    # ./data/data_[12, 24, 48]/[train, val]/partials

    images_path = os.path.join(wider_path, f'{dirname}/images')
    labels_file = os.path.join(wider_path, f'{dirname}/labels.txt')

    report = {'positives':0, 'negatives':0, 'partials':0}

    # Generate output directories
    for dir in [root, outdir, path, positives, negatives, partials]:
        if not os.path.exists(dir):
            print('INFO: generating directory \'', dir, '\'...')
            os.mkdir(dir)
        else:
            print('INFO: directory \'', dir, '\' already exists.')
    
    # Create empty labels file for every generated directory
    files = {}
    for dir in [positives, negatives, partials]:
        labels_path = os.path.join(dir, 'labels.txt')
        print('INFO: creating empty labels file \'', labels_path, '\'...')
        file = open(labels_path, 'w')
        files[dir] = file

    # Read labels
    labels = parse_labels(labels_file)

    tot_negatives = 0  # number of generated negative samples
    tot_positives = 0  # number of generated positive samples
    tot_partials  = 0  # number of generated partial samples
    if n_images < 0:
        n_images = len(labels)
    # Iterate over dataset images
    for label in tqdm.tqdm(labels[:n_images]):
        image_path = os.path.join(images_path, label['path'])
        image = cv2.imread(image_path)
        image_basename = os.path.basename(image_path)[:-4]
        bboxes = np.array(label['bbox'], dtype=np.float32)

        im_h, im_w, _ = image.shape # image is transposed

        n_neg = generate_negatives(image, bboxes, files[negatives], negatives, image_basename, size, verbose=verbose)
        tot_negatives += n_neg

        n_pos, n_part = generate_positives(image, bboxes, [files[positives], files[partials]], [positives, partials],
         image_basename, size, verbose=verbose)
        
        tot_positives += n_pos
        tot_partials += n_part
    
    for file in files.values():
        file.close()
    
    report['positives'] = tot_positives
    report['negatives'] = tot_negatives
    report['partials'] = tot_partials

    return report
    
def print_report(report, name, size):
    negatives = report['negatives']
    positives = report['positives']
    partials = report['partials']
    tot = negatives + positives + partials
    print(f'--- {tot} {name.capitalize()} IMAGES GENERATED ---')
    path = os.path.join(DATA_PATH, f'data_{size}')
    outdir = os.path.join(path, name)
    print(f'path: {outdir}')
    print(f'{positives} positive, {partials} partials and {negatives} negatives were generated')


def main(args):
    imsize = args.imsize
    assert imsize in [12, 24, 48, 64], 'imsize must be in [12, 24, 48, 64]'

    n_images = args.n_images
    verbose = args.verbose
    widerpath = args.widerpath
    datapath = args.datapath
    
    report_train = generate_data(widerpath, datapath, imsize, train=True, n_images=n_images, verbose=verbose)
    report_val = generate_data(widerpath, datapath, imsize, train=False, n_images=n_images, verbose=verbose)

    print_report(report_train, 'train', imsize)
    print_report(report_val, 'val', imsize)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='imsize', type=int, help="Output image size [12, 24, 48, 64]")
    parser.add_argument(dest='widerpath', type=str, help="Path of the widerface dataset")
    parser.add_argument(dest='datapath', type=str, help="Path the generated dataset")
    parser.add_argument('-s', '--size', dest='n_images', type=int, default=-1)
    parser.add_argument('-v', '--verbose', action="store_true", default=False)

    args = parser.parse_args()

    main(args)

