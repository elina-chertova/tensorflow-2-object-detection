import numpy as np
from matplotlib import pyplot as plt
import imageio
import cv2
import tensorflow as tf
import pandas as pd
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
import os

# jpgnames = []
# root_dir = 'images_data/test'
# for entry in os.listdir(root_dir):
#     if os.path.isfile(os.path.join(root_dir, entry)):
#         if entry.split('.')[1] in ['jpeg', 'jpg']:
#             jpgnames.append('images_data/test/' + entry)



class Inference:
    def __init__(self, images_list, saved_model_path, label_map_path):
        self.images_list = images_list
        self.detect_fn = tf.saved_model.load(saved_model_path)
        self.category_index = label_map_util.create_category_index_from_labelmap(label_map_path, use_display_name=True)

    def get_coordinates(self, threshold, output_dict, image_width, image_height):
        output = []

        for index, score in enumerate(output_dict['detection_scores']):
            if score < threshold:
                continue
            label = self.category_index[output_dict['detection_classes'][index]]['name']
            ymin, xmin, ymax, xmax = output_dict['detection_boxes'][index]
            output.append([label, int(xmin * image_width), int(ymin * image_height), int(xmax * image_width),
                           int(ymax * image_height)])
        return output

    def load_image_into_numpy_array(self, image_path):
        """

        :param str path:
        :return: Массив изображения
        """
        return np.array(Image.open(image_path))

    def detection_result(self):
        coordinates = pd.DataFrame({'image': [], 'output': []})
        for k, image_path in enumerate(self.images_list):
            print('Running inference for {}... '.format(image_path), end='')
            image_np = self.load_image_into_numpy_array(image_path)
            print(image_np.shape)
            try:
                image_height, image_width, _ = image_np.shape
            except ValueError:
                image_height, image_width = image_np.shape
            try:
                input_tensor = tf.convert_to_tensor(image_np)
                input_tensor = input_tensor[tf.newaxis, ...]
                detections = self.detect_fn(input_tensor)
                num_detections = int(detections.pop('num_detections'))
                detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
                detections['num_detections'] = num_detections
                print("detections['num_detections'] ", detections['num_detections'])
                detections['detection_classes'] = detections['detection_classes'].astype(np.int64)
                print("detections['detection_classes'] ", detections['detection_classes'], len(detections['detection_classes']))

                image_np_with_detections = image_np.copy()
                vis_util.visualize_boxes_and_labels_on_image_array(
                      image_np_with_detections,
                      detections['detection_boxes'],
                      detections['detection_classes'],
                      detections['detection_scores'],
                      self.category_index,
                      use_normalized_coordinates=True,
                      max_boxes_to_draw=100,
                      min_score_thresh=.5,
                      agnostic_mode=False)
                print('vis_util ', vis_util)
                # plt.figure()
                # plt.imshow(image_np_with_detections)
                # cv2.imwrite(f'pistol{k}inf.jpg', image_np_with_detections)
                # print("detections['detection_boxes'] ", detections['detection_boxes'], len(detections['detection_boxes']),
                #       detections['detection_boxes'][0], type(detections['detection_boxes'][0]))
                print('Done')
                out = self.get_coordinates(0.5, detections, image_width, image_height)
                # print('output =', )
                if len(out) != 0:
                    coordinates.loc[len(coordinates)] = [image_path, out]
                plt.show()
            except ValueError:
                continue
        coordinates.to_csv('result/outputs.csv')


# im = Inference(jpgnames, 'images_data/output/frozen/saved_model', "annotations/label_map.pbtxt")
# im.detection_result()


