"""Flask-based service that detects ads in screenshots."""

import json
import logging
from timeit import default_timer as timer

import flask
import numpy as np
import paste.translogger as tl
from PIL import Image
import waitress

from yolo_v3 import non_max_suppression
from utils import *


TFF = tf.app.flags

TFF.DEFINE_string('class_names', 'cfg/ad.names', 'File with class names')
TFF.DEFINE_string('weights_file', '../models/page_based_yolov3.weights',
                  'Binary file with detector weights')

TFF.DEFINE_integer('size', 416, 'Image size')

TFF.DEFINE_float('conf_threshold', 0.5, 'Confidence threshold')
TFF.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')
TFF.DEFINE_float('match_threshold', 0.4,
                 'IoU required for for detection to be counted as a match')

FLAGS = TFF.FLAGS


# Region type (a.k.a. class) that means "advertisement".
AD_TYPE = 0


def scale_box(box, img_size):
    """Scale detected box to match image size."""
    xscale = img_size[0] / FLAGS.size
    yscale = img_size[1] / FLAGS.size
    x0, y0, x1, y1 = box
    return [
        float(x0) * xscale,
        float(y0) * yscale,
        float(x1) * xscale,
        float(y1) * yscale,
    ]


class AdDetector:
    """Ad detector that encapsulates TF session and detection model."""

    def __init__(self):
        classes = load_coco_names(FLAGS.class_names)
        self.inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])
        config = tf.ConfigProto()
        logging.info('Initializing TF session')
        self.sess = tf.Session(config=config)
        logging.info('Loading YOLOv3 weights')
        self.detections, self.boxes = init_yolo(
            self.sess, self.inputs, len(classes),
            FLAGS.weights_file, header_size=4,
        )
        logging.info('Done')

    def detect(self, image):
        """Detect ads in the image, return detection results as a dict.

        The return value is as follows:

            {
                'size': [image_width, image_height],
                'boxes': [
                    [x0, y0, x1, y1, probability],
                    ...
                ],
            }

        """
        img = image.resize((FLAGS.size, FLAGS.size))
        if img.mode == 'RGBA':
            img = img.convert(mode='RGB')

        logging.info('Detecting ads')
        t1 = timer()
        detected_boxes = self.sess.run(
            self.boxes,
            feed_dict={self.inputs: [np.array(img, dtype=np.float32)]},
        )
        unique_boxes = non_max_suppression(
            detected_boxes,
            confidence_threshold=FLAGS.conf_threshold,
            iou_threshold=FLAGS.iou_threshold,
        )
        boxes = [scale_box(box, image.size) + [float(p)]
                 for box, p in unique_boxes[AD_TYPE]]
        t2 = timer()
        logging.debug('Detected boxes: {}'.format(boxes))
        logging.info('Detection complete: found {} ads in {} seconds'
                     .format(len(boxes), t2 - t1))

        return {
            'size': image.size,
            'boxes': boxes,
            'detection_time': t2 - t1,
        }


app = flask.Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return """
<html>
  <body>
    <form action="/detect" method="POST" enctype="multipart/form-data">
      <input type="file" name="image" />
      <input type="submit" value="submit" name="submit" />
    </form>
  </body>
</html>
"""


@app.route('/detect', methods=['POST'])
def detect():
    image_file = flask.request.files['image']
    image = Image.open(image_file)
    response_body = json.dumps(app.detector.detect(image))
    response_headers = {
        'Content-type': 'application/json',
    }
    return response_body, response_headers


def serve(argv):
    app.detector = AdDetector()
    waitress.serve(tl.TransLogger(app, setup_console_handler=False),
                   listen='*:8080')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    tf.app.run(main=serve)
