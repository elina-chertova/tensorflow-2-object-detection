# Specify model imports
from object_detection.builders import model_builder
from object_detection.utils import config_util
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
import cv2
import numpy as np
import os
import tensorflow as tf

# Disable GPU if necessary
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import argparse
import tensorflow as tf
import cv2

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

# patch tf1 into `utils.ops`
utils_ops.tf = tf.compat.v1

# Patch the location of gfile
tf.gfile = tf.io.gfile


def load_model(model_path):
    """
    :param str model_path: путь к модели
    :return: загруженная модель
    """
    model = tf.saved_model.load(model_path)
    return model


def run_inference_for_single_image(model, image):
    """
    Инференс для одного кадра
    :param model: загруженная модель
    :param image: изображение
    :return: output_dict
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

    #
    if 'detection_masks' in output_dict:
        #
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            output_dict['detection_masks'], output_dict['detection_boxes'],
            image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
        output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

    return output_dict


def run_inference(model, category_index, cap):
    """

    :param model: модель
    :param category_index: индекс
    :param cap: cv2.VideoCapture

    """
    while cap.isOpened():
        ret, image_np = cap.read()
        if not ret:
            break

        #
        output_dict = run_inference_for_single_image(model, image_np)
        #
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8) #,
            # min_score_thresh=0.7
        cv2.imshow('object_detection', cv2.resize(image_np, (850, 600)))
        if cv2.waitKey(50) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


def launch_video(model, labelmap, video_path):
    detection_model = load_model(model)
    category_index = label_map_util.create_category_index_from_labelmap(labelmap, use_display_name=True)

    cap = cv2.VideoCapture(video_path)
    run_inference(detection_model, category_index, cap)


def launch(model, labelmap, video_path):
    """Запуск видео с задетектированными объектами"""
    detection_model = load_model(model)
    category_index = label_map_util.create_category_index_from_labelmap(labelmap, use_display_name=True)

    cap = cv2.VideoCapture(video_path)
    run_inference(detection_model, category_index, cap)

    # parser = argparse.ArgumentParser(description='Detect objects inside webcam videostream')
    # parser.add_argument('-m', '--model', type=str, required=True, help='Model Path')
    # parser.add_argument('-l', '--labelmap', type=str, required=True, help='Path to Labelmap')
    # parser.add_argument('-v', '--video_path', type=str, required=True, help='Path to video.')
    # args = parser.parse_args()
    #
    # detection_model = load_model(args.model)
    # category_index = label_map_util.create_category_index_from_labelmap(args.labelmap, use_display_name=True)
    #
    # cap = cv2.VideoCapture(args.video_path)
    # run_inference(detection_model, category_index, cap)

