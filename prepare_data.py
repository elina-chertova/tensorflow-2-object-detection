import os
import random
import shutil
import glob
from skimage import io
import cv2


def divide_test_and_train(root_dir):
    """
    Создает папки test и train и рандомно раскидывает по ним изображения и соответствующие xml файлы
    :param root_dir: название папки с изображениями
    """
    os.makedirs(root_dir + '/train')
    os.makedirs(root_dir + '/test')
    xmlnames = []
    jpgnames = []
    for entry in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, entry)):
            if entry.split('.')[1] in ['jpeg', 'jpg', 'png']:
                jpgnames.append(entry)
                xmlnames.append(entry.split('.')[0] + '.xml')

    shuffle_list = []
    for jpg, xml in zip(jpgnames, xmlnames):
        print(jpg, xml)
        shuffle_list.append([jpg, xml])

    print('shuffle_list =', shuffle_list)
    random.shuffle(shuffle_list)
    jpgnames = []
    xmlnames = []
    for element in shuffle_list:
        jpgnames.append(element[0])
        xmlnames.append(element[1])

    split = int(0.8 * len(jpgnames))

    file_train = [idx for idx in jpgnames[:split]]
    file_test = [idx for idx in jpgnames[split:]]
    # print('file_train', sorted(file_train))
    xml_train = [idx for idx in xmlnames[:split]]
    xml_test = [idx for idx in xmlnames[split:]]
    # print('xml_train', sorted(xml_train))
    for name in file_train:
        shutil.copy(root_dir + '/' + name, root_dir + "/train/")
    for name in file_test:
        shutil.copy(root_dir + '/' + name, root_dir + "/test/")

    for name in xml_train:
        shutil.copy(root_dir + '/' + name, root_dir + "/train/")
    for name in xml_test:
        shutil.copy(root_dir + '/' + name, root_dir + "/test/")


image_dir = 'images_data'


def remove_unreadable_images(directory_with_images):
    """
    Удаляет поврежденные изображения формата jpg/jpeg и соотвествующие xml
    :param directory_with_images: название папки с изображениями
    """
    files_to_remove = []
    files = glob.glob(directory_with_images + '/*.jpeg') + glob.glob(directory_with_images + '/*.jpg') + glob.glob(directory_with_images + '/*.png')
    for i in range(len(files)):
        try:
            _ = io.imread(files[i])
            img = cv2.imread(files[i])

        except Exception as e:
            print(e)
            files_to_remove.append(files[i])

    # print(files_to_remove)
    for name in files_to_remove:
        if os.path.isfile(name):
            os.remove(name)
            print(f"File {name} deleted (unreadable)")
        else:
            print(f"File {name} doesn't exists!")

        if os.path.isfile(name.split('.')[0] + '.xml'):
            os.remove(name)
            print(f"File {name.split('.')[0] + '.xml'} success")
        else:
            print(f"File {name.split('.')[0] + '.xml'} doesn't exists!")


divide_test_and_train(image_dir)
remove_unreadable_images(image_dir)
