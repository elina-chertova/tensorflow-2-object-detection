import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage
import subprocess
import tensorflow as tf
from tensorboard import program
from IPython import get_ipython
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
# from object_detection.utils import colab_utils
from object_detection.builders import model_builder
from subprocess import run, STDOUT, PIPE
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET
from PIL import Image
import re
from collections import namedtuple, OrderedDict
import sys

sys.path.insert(0, 'models/research')

from object_detection.utils import dataset_util
import io
import time

choose_models = pd.read_csv('All_Models', index_col=False)


class ObjectDetection:
    def __init__(self, model_number=20):
        """
        path_to_label_map_pbtxt -
        path_to_mydata -
        """
        super().__init__()
        # self.annotations = annotations
        self.model_number = model_number
        self.path_to_directory = os.getcwd()
        self.path_to_annotations = self.path_to_directory + '/annotations'
        self.path_to_images = self.path_to_directory + '/images_data'
        self.path_to_label_map_pbtxt = self.path_to_annotations + '/label_map.pbtxt'
        self.path_to_config = self.path_to_images + '/' + \
                              choose_models.iloc[self.model_number]['Link'].split('/')[-1].split('.')[
                                  0] + '/pipeline.config'
        self.path_to_train_data = self.path_to_images + '/train'
        self.path_to_test_data = self.path_to_images + '/test'

    def label_map(self, annotations):
        """
        Создает файл с описанием классов формата .pbtxt
                Параметры:
                        annotations (list): список классов для детектирования
        """
        with open(self.path_to_label_map_pbtxt, 'a') as file_:
            for id_ in range(1, len(annotations) + 1):
                list_of_string = ['item\n', '{\n', '  id: {}'.format(int(id_)), '\n',
                                  "  name:'{0}'".format(str(annotations[id_ - 1])), '\n', '}\n']
                for string in list_of_string:
                    file_.write(string)

    def xml_to_csv(self, path):
        """
        Создает датафрейм из имеющихся в папке /images_data файлов формата .xml
                Параметры:
                        path (str): путь до папки с xml файлами
                Возвращаемое значение:
                        xml_df (DataFrame): датафрейм из данных о размеченных изображениях
        """
        xml_list = []
        for xml_file in glob.glob(path + '/*.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
        column_name = ['id', 'width', 'height', 'class_', 'xmin', 'ymin', 'xmax', 'ymax']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        return xml_df

    def create_annot_csv(self, path_to_dataset):
        """
        Создает и возвращает датафрейм, содержащий информацию о датасете из изображений.
        В случае, если csv с информауией уже существует, возвращает датафрейм с ним. Иначе - создает его из xml-файлов.
                Параметры:
                        path_to_dataset (str): путь к папке /images_data/all_images_data

                Возвращаемое значение:
                        annot/xml_df (DataFrame): датафрейм с информацией обо всех объектах датасета
        """
        extension = path_to_dataset[-3:]
        if extension == 'csv':
            annot = pd.read_csv(path_to_dataset)
            return annot
        else:
            xml_df = self.xml_to_csv(path_to_dataset)
            try:
                xml_df.to_csv(self.path_to_images + '/data.csv', index=None)
            except:
                path_temp = self.path_to_images + '/data.csv'
                xml_df.to_csv('/'.join(path_temp.split('/')[-2:]), index=None)
            return xml_df

    def write_to_record(self, annot, annotations):  # path_to_mydata
        """
        Преобразование аннотаций в формат  TFRecord для обучения с помощью Tensorflow Object Detection
                Параметры:
                        annot (DataFrame): датафрейм с информацией обо всех объектах датасета
                        annotations (list): список из классов для детектирования
        """

        def split(df, group):
            data = namedtuple('data', ['id', 'object'])
            gb = df.groupby(group)
            return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

        train = os.listdir(self.path_to_images + '/train')
        test = os.listdir(self.path_to_images + '/test')
        df_train = annot.loc[~annot.id.isin(test)]
        df_test = annot.loc[~annot.id.isin(train)]
        writer = tf.io.TFRecordWriter(self.path_to_annotations + '/train_data.record')

        grouped_train = split(df_train, 'id')
        for group in grouped_train:
            tf_example = self.create_tf_example(group, self.path_to_train_data, annotations)  # 'cup_book/'
            writer.write(tf_example.SerializeToString())
        writer.close()

        writer = tf.io.TFRecordWriter(self.path_to_annotations + '/test_data.record')
        grouped_test = split(df_test, 'id')
        for group in grouped_test:
            tf_example = self.create_tf_example(group, self.path_to_test_data, annotations)  # 'cup_book/'
            writer.write(tf_example.SerializeToString())
        writer.close()
        # writer = tf.python_io.TFRecordWriter(path_to_mydata + 'train_data.record')

        # for idx, row in annot.iterrows():
        #     tf_example = create_tf_example(row)
        #     writer.write(tf_example.SerializeToString())

        # writer.close()

    def class_text_to_int(self, row_label, d):
        """
        Возвращает номер класса
                Параметры:
                        row_label (str): название класса
                        d (dict): словарь из классов и их нумерации
                Возвращаемое значение:
                        d[row_label] : номер класса
        """
        return d[row_label]

    def create_tf_example(self, group, path, annotations):
        """
        Функция для создания tf_example (формат данных для тензорфлоу) из датасета
                Параметры:
                        group (__main__.data): строки из датафрейма для создания формата для object detection
                        path (str): путь к папке /images_data
                        annotations (list): список классов для детектирования
                Возвращаемое значение:
                        tf_example (tf.train.Example): формат хранения данных для обучения и инференса
        """
        d = {k: v for v, k in enumerate(annotations, start=1)}
        with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.id)), 'rb') as fid:
            encoded_image_data = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_image_data)
        img = Image.open(encoded_jpg_io)
        filename = group.id.encode('utf8')
        # img_fpath = os.path.join('cup_book', example.id)  # /content/drive/MyDrive/tf_book/
        # img = Image.open(img_fpath)
        height = img.size[1]
        width = img.size[0]
        # filename = str.encode(example.id)
        # with open(img_fpath, mode='rb') as f:
        #     encoded_image_data = f.read()
        image_format = b'jpeg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class_'].encode('utf8'))
            classes.append(self.class_text_to_int(row['class_'], d))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_image_data),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example

    def copy_pipeline(self):
        """ Чтение данных из папки выбранной модели и запись их в переменную s """
        with open(
                self.path_to_directory + '/' + choose_models.iloc[self.model_number]['Link'].split('/')[-1].split('.')[
                    0] + '/pipeline.config', 'r') as f:
            s = f.read()
        return s

    def create_pipeline_config(self, s, annotations):
        """
        Создание файла pipeline.config и замена необходимых строк на необходимые для конкретной задачи параметры.
                Параметры:
                        s (str): содержимое файла pipeline.config
                        annotations (list): список классов для детектирования
        """
        config = self.path_to_directory + '/' + choose_models.iloc[self.model_number]['Link'].split('/')[-1].split('.')[
            0] + '/pipeline.config'
        print('config', config)
        fine_tune_checkpoint = '/'.join(config.split('/')[:-1]) + '/checkpoint/ckpt-0'  # '/model.ckpt'
        print('fine_tune_checkpoint', fine_tune_checkpoint)
        train_record = self.path_to_annotations + '/train_data.record'
        test_record = self.path_to_annotations + '/test_data.record'
        print('train_record', train_record)
        label_map_pbtxt_fname = self.path_to_label_map_pbtxt
        print('label_map_pbtxt_fname', label_map_pbtxt_fname)
        batch_size = 12
        num_classes = len(annotations)
        num_steps = 200000
        print(self.path_to_images + '/pipeline.config')

        with open(self.path_to_images + '/pipeline.config', 'w') as f:
            f.write(s)

        # tf.keras.backend.clear_session()

        # pipeline_config_im_data = self.path_to_images + '/pipeline.config'
        # "/content/models/research/object_detection/configs/tf2/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8.config"
        # configs = config_util.get_configs_from_pipeline_file(pipeline_config_im_data)
        #
        with open(self.path_to_images + '/pipeline.config', 'w') as f:
            # fine_tune_checkpoint
            s = re.sub('fine_tune_checkpoint: ".*?"',
                       'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), s)

            # tfrecord files train and test.
            s = re.sub(
                'input_path: ".*?"', 'input_path: "{}"'.format(test_record), s)
            s = re.sub(
                'input_path: ".*?"', 'input_path: "{}"'.format(train_record), s, 1)

            # label_map_path
            s = re.sub(
                'label_map_path: ".*?"', 'label_map_path: "{}"'.format(label_map_pbtxt_fname), s)

            # Set training batch_size.
            s = re.sub('batch_size: [0-9]+',
                       'batch_size: {}'.format(batch_size), s)

            # Set training steps, num_steps
            s = re.sub('num_steps: [0-9]+',
                       'num_steps: {}'.format(num_steps), s)

            # Set number of classes num_classes.
            s = re.sub('num_classes: [0-9]+',
                       'num_classes: {}'.format(num_classes), s)

            s = re.sub('use_dropout: false',
                       'use_dropout: true', s)

            s = re.sub('freeze_batchnorm: false',
                       'freeze_batchnorm: true', s)
            # fine-tune checkpoint type
            s = re.sub(
                'fine_tune_checkpoint_type: "classification"', 'fine_tune_checkpoint_type: "{}"'.format('detection'), s)
            f.write(s)

    def __call__(self):
        """
        """
        if tf.test.gpu_device_name():
            print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
        else:
            print("Please install GPU version of TF")

        os.system('wget {}'.format(choose_models.iloc[self.model_number]['Link']))
        os.system('tar -xzf {}'.format(choose_models.iloc[self.model_number]['Link'].split('/')[-1]))
        time.sleep(15)
        df = self.create_annot_csv(self.path_to_images + '/all_images_data')
        annotations = list(set(df['class_']))
        print('annotations ==', annotations)
        time.sleep(5)
        self.label_map(annotations)
        # print(df)
        time.sleep(5)
        self.write_to_record(df, annotations)
        time.sleep(5)
        s = self.copy_pipeline()
        time.sleep(5)
        self.create_pipeline_config(s, annotations)
        time.sleep(15)
        tracking_address = 'images_data/output'
        tb = program.TensorBoard()
        tb.configure(argv=[None, '--logdir', tracking_address])
        url = tb.launch()
        print(f"Tensorflow listening on {url}")

        return 0


detect = ObjectDetection()
detect()