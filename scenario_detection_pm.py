import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import os
import random
import zipfile
import io
import scipy.misc
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import cv2
import os
from object_detection.utils import label_map_util
import pathlib
import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2
import tensorflow as tf
import pandas as pd
import sys
# sys.path.insert(0, 'models/research')
from PIL import Image
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import warnings

MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8'
TF_MODELS_BASE_PATH = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/'
CACHE_FOLDER = './cache'


# PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

class ScenarioDetector:
    def __init__(self, model_name='centernet_hg104_1024x1024_coco17_tpu-32', model_date='20200711',
                 sequence_of_actions=None, images_list=None, label_filename='mscoco_label_map.pbtxt'):
        """

        :param model_name: название модели
        :param model_date: дата создания модели
        :param sequence_of_actions: последовательность действий
        :param images_list: список из изображений
        :param label_filename: путь к файлу с метками
        """
        self.model_name = model_name
        self.model_date = model_date
        self.label_filename = label_filename
        self.sequence_of_actions = sequence_of_actions
        self.PATH_TO_MODEL_DIR = self.download_model()
        self.PATH_TO_SAVED_MODEL = self.PATH_TO_MODEL_DIR + "/saved_model"
        self.detect_fn = tf.saved_model.load(self.PATH_TO_SAVED_MODEL)
        self.PATH_TO_LABELS = self.download_labels()
        self.images_list = images_list
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS,
                                                                                 use_display_name=True)

    def download_model(self):
        """
        Загружает модель.
        :return: путь до загруженной модели
        """
        base_url = 'http://download.tensorflow.org/models/object_detection/tf2/'
        model_file = self.model_name + '.tar.gz'
        model_dir = tf.keras.utils.get_file(fname=self.model_name,
                                            origin=base_url + self.model_date + '/' + model_file,
                                            untar=True)
        return str(model_dir)

    def download_labels(self):
        """
        Загружает файл с объектами и метками.
        :return: путь до файла с метками
        """
        base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/data/'
        label_dir = tf.keras.utils.get_file(fname=self.label_filename,
                                            origin=base_url + self.label_filename,
                                            untar=False)
        label_dir = pathlib.Path(label_dir)
        return str(label_dir)

    def run_inference_for_single_image(self, model, image):
        """
        Инференс для одного кадра.
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

        if 'detection_masks' in output_dict:
            detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                output_dict['detection_masks'], output_dict['detection_boxes'],
                image.shape[0], image.shape[1])
            detection_masks_reframed = tf.cast(detection_masks_reframed > 0.5, tf.uint8)
            output_dict['detection_masks_reframed'] = detection_masks_reframed.numpy()

        return output_dict

    def get_classes_from_frame(self, threshold, output_dict):
        """
        Возвращает список предсказанных классов.
        :param threshold: уровень доверия
        :param output_dict: предсказанные параметры
        :return:
        """
        output = []

        for index, score in enumerate(output_dict['detection_scores']):
            if score < threshold:
                continue
            label = self.category_index[output_dict['detection_classes'][index]]['name']
            output.append(label)
        return output

    def detect(self, image, model, count):
        """
        Детекция изображения.
        :param image: путь к изображению
        :param model: путь к модели
        :param count: номер изображения
        :param label_offset:
        :return:
        """
        output_dict = self.run_inference_for_single_image(model, image)
        #
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            self.category_index,
            instance_masks=output_dict.get('detection_masks_reframed', None),
            use_normalized_coordinates=True,
            line_thickness=8)
        out_ = self.get_classes_from_frame(0.5, output_dict)
        cv2.imwrite(f'output_frames/pistol{count}inf.jpg', image)
        print(f'pistol{count}inf.jpg = ', out_)
        #
        return image

    def load_image_into_numpy_array(self, image_path):
        """
        :param str path:
        :return: Массив изображения
        """
        return np.array(Image.open(image_path))

    def check_sequence(self, class_number, predict_class, res, diff_classes, number_correct_mistakes):
        """
        Проверка корректности последовательности задетектированных объектов.
        :param int class_number: ожидаемый индекс списка последовательности объектов
        :param list predict_class: задетектированные на кадре объекты
        :param list res: список из True/False полученного объекта последовательности
        :param set diff_classes: все предсказанные классы
        :param dict number_correct_mistakes: словарь из классов и количества допустимых ошибок
        :return:
        """
        sequence = self.sequence_of_actions
        diff_classes.update(predict_class)

        old_predict_class = predict_class
        predict_class = [old_predict_class[i] for i in range(len(old_predict_class)) if old_predict_class[i] in sequence]

        if not predict_class:
            res.append(0)
            return res, class_number

        if sequence[class_number] in predict_class:
            res.append(0)
        elif class_number < len(sequence) - 1 and sequence[class_number + 1] in predict_class:
            class_number += 1
            res.append(0)
        else:
            number_correct_mistakes[sequence[class_number]] -= 1
            print(f'mistake in {sequence[class_number]}')
            # res.append(1)
        if number_correct_mistakes[sequence[class_number]] < 0:
            res.append(1)

        return res, class_number

    def run_video(self, path_to_video):
        """
        Трекинг видео.
        :param path_to_video:
        :return: корректность последовательности
        """
        global response
        class_number = 0
        res = []
        diff_classes = set()
        lst = [5 for i in range(len(self.sequence_of_actions))]
        number_correct_mistakes = dict(zip(self.sequence_of_actions, lst))

        model = self.detect_fn
        cap = cv2.VideoCapture(path_to_video)
        # cap.set(3, 640)
        # cap.set(4, 480)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))
        k = 0

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('len =', length)
        while cap.isOpened():
            ret, image_np = cap.read()
            if not ret:
                break
            # cv2.imwrite(f'input_frames/{k}inf.jpg', image_np)
            #
            output_dict = self.run_inference_for_single_image(model, image_np)
            # Визуализация результатов детектирования
            # track_ids = [0, 1, 2]
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                output_dict['detection_boxes'],
                output_dict['detection_classes'],
                output_dict['detection_scores'],
                self.category_index,
                instance_masks=output_dict.get('detection_masks_reframed', None),
                use_normalized_coordinates=True,
                line_thickness=8)
            # cv2.imshow('object_detection', cv2.resize(image_np, (850, 600)))
            cv2.imshow('object_detection', image_np)
            # cv2.imwrite(f'output_frames/pistol{k}inf.jpg', image_np)
            print()
            out.write(image_np)
            k += 1
            out_ = self.get_classes_from_frame(0.5, output_dict)

            res, class_number = self.check_sequence(class_number, out_, res, diff_classes, number_correct_mistakes)
            print(f'pistol{k}inf.jpg = ', out_)
            # cv2.imshow('frame', frame)
            c = cv2.waitKey(1)
            if sum(res) == 0:
                print('ok')
                response = 'ok'
            else:
                response = 'wrong sequence'
                print('wrong sequence')
            if c & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        return response

#
# MODEL_DATE = '20200711'
# MODEL_NAME = 'centernet_hg104_1024x1024_coco17_tpu-32'
# actions = ['laptop', 'cup', 'person']
# sc = ScenarioDetector(MODEL_NAME, MODEL_DATE, actions)
# sc.run_video('scenario/test_video.mp4')
