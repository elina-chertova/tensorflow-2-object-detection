import os
import random
import shutil
import glob
from skimage import io
import cv2
from collections import Counter


class PrepareData:
    def __init__(self, image_folder='images_data', train_set_percent=0.8):
        super().__init__()
        self.train_set_percent = train_set_percent
        self.root_dir = image_folder
        self.test_dir = self.root_dir + '/test'
        self.train_dir = self.root_dir + '/train'

    def create_test_and_train_folder(self):
        """
        Создает тестовую и обучающую папки
        :return:
        """
        try:
            os.makedirs(self.root_dir + '/train')
        except FileExistsError:
            print('Папка train уже создана')
        try:
            os.makedirs(self.root_dir + '/test')
        except FileExistsError:
            print('Папка test уже создана')
        try:
            os.makedirs('annotations')
        except FileExistsError:
            print('Папка annotations уже создана')

        try:
            os.makedirs('result')
        except FileExistsError:
            print('Папка result уже создана')

    def divide_test_and_train(self):
        """
        Создает папки test и train и рандомно раскидывает по ним изображения и соответствующие xml файлы

        """

        xmlnames = []
        jpgnames = []
        for entry in os.listdir(self.root_dir):
            if os.path.isfile(os.path.join(self.root_dir, entry)):
                if entry.split('.')[1] in ['jpeg', 'jpg']:
                    if entry.split('.')[0] + '.xml' in os.listdir(self.root_dir):
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

        split = int(self.train_set_percent * len(jpgnames))

        file_train = [idx for idx in jpgnames[:split]]
        file_test = [idx for idx in jpgnames[split:]]
        # print('file_train', sorted(file_train))
        xml_train = [idx for idx in xmlnames[:split]]
        xml_test = [idx for idx in xmlnames[split:]]
        # print('xml_train', sorted(xml_train))
        for name in file_train:
            try:
                shutil.copy(self.root_dir + '/' + name, self.root_dir + "/train/")
            except FileNotFoundError:
                print(f'File does not exist {self.root_dir + "/" + name}')
        for name in file_test:
            try:
                shutil.copy(self.root_dir + '/' + name, self.root_dir + "/test/")
            except FileNotFoundError:
                print(f'File does not exist {self.root_dir + "/" + name}')

        for name in xml_train:
            try:
                shutil.copy(self.root_dir + '/' + name, self.root_dir + "/train/")
            except FileNotFoundError:
                print(f'File does not exist {self.root_dir + "/" + name}')
        for name in xml_test:
            try:
                shutil.copy(self.root_dir + '/' + name, self.root_dir + "/test/")
            except FileNotFoundError:
                print(f'File does not exist {self.root_dir + "/" + name}')

    def remove_unreadable_images(self):
        """
        Удаляет поврежденные изображения формата jpg/jpeg и соотвествующие xml
        """
        files_to_remove = []
        files = glob.glob(self.root_dir + '/*.jpeg') + glob.glob(self.root_dir + '/*.jpg')
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

    def remove_files_without_pair(self):
        """
        Удаляет изображения, у которых нет аннотаций, и аннотации, у которых нет изображения
        """
        print(os.listdir(self.test_dir))
        print(os.listdir(self.train_dir))
        test_name = []
        for i in os.listdir(self.test_dir):
            test_name.append(i.split('.')[0])
        count_test = Counter(test_name)
        delete_test_files = []
        for k in count_test.keys():
            if count_test[k] != 2:
                delete_test_files.append(k)

        train_name = []
        for i in os.listdir(self.train_dir):
            train_name.append(i.split('.')[0])
        count_train = Counter(train_name)
        delete_train_files = []
        for k in count_train.keys():
            if count_train[k] != 2:
                delete_train_files.append(k)

        for entry in os.listdir(self.test_dir):
            if entry.split('.')[0] in delete_test_files:
                try:
                    os.remove(self.test_dir + '/' + entry)
                    os.remove(self.root_dir + '/' + entry)
                except FileNotFoundError:
                    print(f"File {self.test_dir + '/' + entry} doesn't exists!")

        for entry in os.listdir(self.train_dir):
            if entry.split('.')[0] in delete_train_files:
                try:
                    os.remove(self.train_dir + '/' + entry)
                    os.remove(self.root_dir + '/' + entry)
                except FileNotFoundError:
                    print(f"File {self.train_dir + '/' + entry} doesn't exists!")

        for i in os.listdir(self.test_dir):
            if i.split('.')[1] in ['JPG']:
                try:
                    os.remove(self.test_dir + '/' + i)
                    os.remove(self.root_dir + '/' + i)
                except FileNotFoundError:
                    print(f"File {self.test_dir + '/' + i} doesn't exists!")
                try:
                    os.remove(self.test_dir + '/' + i.split('.')[0] + '.xml')
                    os.remove(self.root_dir + '/' + i.split('.')[0] + '.xml')
                except FileNotFoundError:
                    print(f"File {self.test_dir + '/' + i} doesn't exists!")

        for i in os.listdir(self.train_dir):
            if i.split('.')[1] in ['JPG']:
                try:
                    os.remove(self.train_dir + '/' + i)
                    os.remove(self.root_dir + '/' + i)
                except FileNotFoundError:
                    print(f"File {self.train_dir + '/' + i} doesn't exists!")
                try:
                    os.remove(self.train_dir + '/' + i.split('.')[0] + '.xml')
                    os.remove(self.root_dir + '/' + i.split('.')[0] + '.xml')
                except FileNotFoundError:
                    print(f"File {self.train_dir + '/' + i} doesn't exists!")

    def move_all_images_to_images_directory(self):
        """
        Переносит все изображения и аннотации в /all_images_data
        """

        target_dir = self.root_dir + '/all_images_data'
        try:
            os.makedirs(target_dir)
        except FileExistsError:
            print('Папка /all_images_data уже создана')

        file_names = os.listdir(self.root_dir)
        for file_name in file_names:
            if file_name not in ['train/', 'train', 'test', 'test/']:
                shutil.move(os.path.join(self.root_dir, file_name), target_dir)
        return 0


# cl = PrepareData(image_folder='images_data')
# # cl.create_test_and_train_folder()
# # cl.divide_test_and_train()
# # cl.remove_unreadable_images()
# # # cl.remove_files_without_pair()
# cl.move_all_images_to_images_directory()
# test_dir = 'images_data/test'
# train_dir = 'images_data/train'
# create_test_and_train_folder(image_dir)
# remove_files_without_pair(test_dir, train_dir, image_dir)
# print(glob.glob(image_dir + '/*.jpeg') + glob.glob(image_dir + '/*.jpg'))
# remove_unreadable_images(image_dir)
# divide_test_and_train(image_dir)

# print(os.listdir(image_dir + '/train'))
# print(os.listdir(image_dir + '/test'))
