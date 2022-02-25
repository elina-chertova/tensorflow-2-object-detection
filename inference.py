import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2
import tensorflow as tf

import sys
# sys.path.insert(0, 'models/research')

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


category_index = label_map_util.create_category_index_from_labelmap("annotations/label_map.pbtxt",use_display_name=True)

detect_fn = tf.saved_model.load('images_data/output/frozen/saved_model')
img = ['images_data/IMG_9046.jpeg', 'images_data/IMG_9043.jpeg']


def load_image_into_numpy_array(path):
    return np.array(Image.open(path))


for k, image_path in enumerate(img):
    print('Running inference for {}... '.format(image_path), end='')
    image_np = load_image_into_numpy_array(image_path)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
    image_np_with_detections = image_np.copy()
    vis_util.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=100,
          min_score_thresh=.5,
          agnostic_mode=False)
    plt.figure()
    plt.imshow(image_np_with_detections)
    cv2.imwrite(f'{k}inf.jpg', image_np_with_detections)
    print('Done')
    plt.show()


