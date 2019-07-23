"""Measure the performance of ML ad detector."""

import csv
import glob
import json
import os
from timeit import default_timer as timer

import numpy as np

from yolo_v3 import non_max_suppression
from utils import *


TFF = tf.app.flags

TFF.DEFINE_string('input_dir', '', 'Input directory')
TFF.DEFINE_string('output_dir', '', 'Output directory')
TFF.DEFINE_string('class_names', 'cfg/ad.names', 'File with class names')
TFF.DEFINE_string('weights_file', '../models/page_based_yolov3.weights',
                  'Binary file with detector weights')

TFF.DEFINE_integer('size', 416, 'Image size')

TFF.DEFINE_float('supp_threshold', 0.1, 'Suppression threshold')
TFF.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
TFF.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')
TFF.DEFINE_float('match_threshold', 0.4,
                 'IoU required for for detection to be counted as a match')

FLAGS = TFF.FLAGS


def convert_regions(regions_dict):
    """Convert region coordinates from cx, cy, w, h -> x1, y1, x2, y2."""

    def convert_box(box):
        cx, cy, w, h = box
        return np.array([
            cx - w / 2, cy - h / 2,
            cx + w / 2, cy + h / 2,
        ])

    return {
        r_type: [
            (convert_box(box), p)
            for box, p in regions
        ]
        for r_type, regions in regions_dict.items()
    }


def load_regions(input_dir, region_types=['textad', 'bannerad']):
    """Load regions information from a CSV file."""
    rmap = {}
    for csv_file in glob.glob(os.path.join(input_dir, '*.csv')):
        with open(csv_file, 'r', encoding='utf-8') as f:
            for i, row in enumerate(csv.reader(f)):
                if i == 0 and row[1:5] == ['xmin', 'ymin', 'xmax', 'ymax']:
                    continue  # Header.
                if row[5] not in region_types:
                    continue  # Non-selected region type
                image = os.path.join(input_dir, row[0])
                box = np.array([float(i) for i in row[1:5]])
                rmap.setdefault(image, []).append(box)

    # Rescale box coordinates to be in [0,1] interval and save in the right
    # data structure.
    ret = {}
    for image, boxes in rmap.items():
        img = Image.open(image)
        scaling_factor = np.array(list(img.size) * 2)
        ret[image] = {0: [(box / scaling_factor, 1) for box in boxes]}

    return ret


def load_image_metadata(input_dir):
    """Load the list of images and regions.

    The input directory can contain one of:
    - `images` and `labels` subdirectories with images in the former and region
      description in corresponding text files in the latter (the layout used
      in original ad-versarial datasets), or
    - images directly inside with regions described in one CSV file (the layout
      we used at eyeo so far).

    Parameters
    ----------
    input_dir : str
        Input directory.

    Returns
    -------
    metadata : [(str, {int: [(box, float)]})]
        Image data records that contain the path to the image followed by
        the map of regions that maps region types to boxes and probabilities.
        Boxes are numpy arrays of floats (x1, y1, x2, y2) where the coordinates
        are rescaled to be between 0 and 1.

    """
    img_dir = os.path.join(input_dir, 'images')
    if os.path.isdir(img_dir):
        image_files = get_images(img_dir)
        image_names = [get_file_name(f) for f in image_files]
        label_files = [
            os.path.join(input_dir, 'labels', name + '.txt')
            for name in image_names
        ]
        return [
            (image_file, convert_regions(load_labels(label_file)))
            for image_file, label_file in zip(image_files, label_files)
        ]
    else:
        regions = load_regions(input_dir)
        return [
            (image_file, regions.get(image_file, {0: []}))
            for image_file in get_images(input_dir)
        ]


def scale_regions(regions_dict, ratio):
    """Scale all regions in the region dict by `ratio`."""
    return {
        r_type: [
            (box * ratio, p)
            for box, p in regions
        ]
        for r_type, regions in regions_dict.items()
    }


def iou(box1, box2):
    # Sadly _iou in yolo_v3.py is protected so we just replicate it here.
    # TODO: Turns out it's also broken for our purposes because it doesn't
    # handle nonoverlapping boxes right.

    b1_x0, b1_y0, b1_x1, b1_y1 = box1
    b2_x0, b2_y0, b2_x1, b2_y1 = box2

    int_x0 = max(b1_x0, b2_x0)
    int_y0 = max(b1_y0, b2_y0)
    int_x1 = min(b1_x1, b2_x1)
    int_y1 = min(b1_y1, b2_y1)

    if int_x0 > int_x1 or int_y0 > int_y1:  # No intersection.
        return 0

    int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

    b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
    b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

    # we add small epsilon of 1e-05 to avoid division by 0
    iou = int_area / (b1_area + b2_area - int_area + 1e-05)
    return iou


def compare(detected, expected):
    """Compare detected boxes to expected boxes.

    Returns a tuple: (true_positives, false_negatives, false_positives) where:
    - true_positives is the number of expected regions that were detected,
    - false_negatives is the number of expected regions that were not detected,
    - false_positives is the number of detected regions that were not expected.

    """
    detected_boxes = [tuple(b) for _, rs in detected.items() for b, _ in rs]
    expected_boxes = [tuple(b) for _, rs in expected.items() for b, _ in rs]
    matches = {}
    for eb in expected_boxes:
        for db in detected_boxes:
            if iou(eb, db) > FLAGS.match_threshold:
                matches[eb] = db
    tp = len(matches)
    fn = len(expected_boxes) - tp
    fp = len(detected_boxes) - tp
    return tp, fn, fp


def finalize_summary(summary, path):
    """Finalize the summary: calculate totals and save as JSON."""
    stats = {
        s: sum(i[s] for i in summary['images'])
        for s in ['tp', 'fn', 'fp']
    }
    stats['recall'] = stats['tp'] / (stats['tp'] + stats['fn'])
    stats['precision'] = stats['tp'] / (stats['tp'] + stats['fp'])
    summary['stats'] = stats

    with open(path, 'wt', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    print('\nOverall results:')
    print('TP:{0[tp]} FN:{0[fn]} FP:{0[fp]}'.format(stats))
    print('Recall: {0[recall]:.2%}'.format(stats))
    print('Precision: {0[precision]:.2%}'.format(stats))


def conv_boxes(box_map):
    """Convert boxes for category 0 to JSON-friendly format."""
    return [
        {
            'x0': float(box[0]),
            'y0': float(box[1]),
            'x1': float(box[2]),
            'y1': float(box[3]),
            'p': float(p),
        }
        for box, p in box_map.get(0, [])
    ],


def main(argv):
    classes = load_coco_names(FLAGS.class_names)
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])
    config = tf.ConfigProto()
    sess = tf.Session(config=config)
    detections, boxes = init_yolo(
        sess, inputs, len(classes),
        FLAGS.weights_file, header_size=4,
    )
    image_meta = load_image_metadata(FLAGS.input_dir)
    safe_mkdir(FLAGS.output_dir)
    summary = {
        'flags': {
            'input_dir': FLAGS.input_dir,
            'output_dir': FLAGS.output_dir,
            'size': FLAGS.size,
            'suppression_threshold': FLAGS.supp_threshold,
            'detection_threshold': FLAGS.conf_threshold,
            'iou_threshold': FLAGS.iou_threshold,
            'match_threshold': FLAGS.match_threshold,
        },
        'images': [],
    }

    for idx, (image_file, regions) in enumerate(image_meta):
        in_name = os.path.basename(image_file)
        out_name = '{}.png'.format(idx)
        print(in_name, '->', out_name)

        img_orig = Image.open(image_file)
        img = img_orig.resize((416, 416))
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')

        t1 = timer()
        detected_boxes = sess.run(
            boxes,
            feed_dict={inputs: [np.array(img, dtype=np.float32)]},
        )
        t2 = timer()
        unique_boxes = non_max_suppression(
            detected_boxes,
            confidence_threshold=FLAGS.supp_threshold,
            iou_threshold=FLAGS.iou_threshold,
        )
        filtered_boxes = {
            rtype: [
                (box, p)
                for box, p in regions
                if p > FLAGS.conf_threshold
            ]
            for rtype, regions in unique_boxes.items()
        }
        scaled_regions = scale_regions(regions, FLAGS.size)
        tp, fn, fp = compare(filtered_boxes, scaled_regions)
        t3 = timer()

        print('\ttotal time: {}'.format(t3 - t1))
        print('\tTP:{} FN:{} FP:{} Recall:{:.2%} Precision:{:.2%}'
              .format(tp, fn, fp, tp / (tp + fn + 1e-5), tp / (tp + fp + 1e-5)))

        draw_boxes(scaled_regions, img_orig, classes, (FLAGS.size, FLAGS.size),
                   (0, 255, 0))
        draw_boxes(filtered_boxes, img_orig, classes, (FLAGS.size, FLAGS.size))
        img_orig.save(os.path.join(FLAGS.output_dir, out_name))

        summary['images'].append({
            'in_name': in_name,
            'out_name': out_name,
            'nn_time': t2 - t1,
            'total_time': t3 - t1,
            'tp': tp,
            'fn': fn,
            'fp': fp,
            'detected_boxes': conv_boxes(unique_boxes),
            'boxes_above_threshold': conv_boxes(filtered_boxes),
            'marked_boxes': conv_boxes(scaled_regions),
        })

    finalize_summary(
        summary,
        os.path.join(FLAGS.output_dir, 'summary.json'),
    )


if __name__ == '__main__':
    tf.app.run()
