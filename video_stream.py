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
import os

warnings.filterwarnings('ignore')
directory_path = os.path.dirname(os.path.abspath(__file__))
data_csv = os.path.join(directory_path, "annotations/label_map.pbtxt")  # "annotations/label_map.pbtxt"
category_index = label_map_util.create_category_index_from_labelmap(data_csv, use_display_name=True)


directory_path = os.path.dirname(os.path.abspath(__file__))
data_csv = os.path.join(directory_path, 'images_data/output/frozen/saved_model')
detection_model = tf.saved_model.load(data_csv)
# img = ['mydata/test3.jpeg', 'mydata/test1.jpeg']

cap = cv2.VideoCapture(0)  # or cap = cv2.VideoCapture("<video-path>")


def run_inference_for_single_image(model, image):
    """

    :param model: Путь к модели
    :param image: Изображение
    :return: Словарь с наименованием класса и его положением на изображении
    """
    image = np.asarray(image)
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    output_dict = model(input_tensor)

    num_detections = int(output_dict.pop('num_detections'))
    output_dict = {key: value[0, :num_detections].numpy()
                   for key, value in output_dict.items()}
    output_dict['num_detections'] = num_detections
    output_dict['detection_classes'] = output_dict['detection_classes'].astype(np.int64)

    if 'detection_masks' in output_dict:
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, cap):
    """Запускает видеокамеру с обнаружением объектов.

    :param model: Путь к модели
    :param cap: cv2.VideoCapture(0)
    """
    while cap.isOpened():
        ret, image_np = cap.read()
        output_dict = run_inference_for_single_image(model, image_np)
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        cv2.imshow('object_detection', cv2.resize(image_np, (800, 600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# run_inference(detection_model, cap)
