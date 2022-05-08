# Документация по детектированию объектов и действий

Требования 1
------------

* Nvidia GPU (GTX 650 or newer)

* CUDA Toolkit v11.2

* CuDNN 8.1.0

* CUDA Toolkit - Anaconda Python 3.7

* OS - Windows, Linux

Требования 2
------------

* Python - 3.7

* TensorFlow >= 2.5, <=2.8



Установка необходимых модулей
-----------------------------
Клонирование ветки репозитория с моделями Tensorflow
```shell script
    git clone https://github.com/tensorflow/models.git
```

Репозиторий
-----------
Ссылка на репозиторий с кодом и зависимостями - https://github.com/elina-chertova/tensorflow-2-object-detection

```shell script
    pip3 install -r requirements.txt
```
Установка Object Detection
--------------------------

```shell script
    cd models/research/
    protoc object_detection/protos/*.proto --python_out=.
    cp object_detection/packages/tf2/setup.py .
    python -m pip install .
```


Тестирование установки Object Detection
---------------------------------------
```shell script
    python models/research/object_detection/builders/model_builder_tf2_test.py
```

Начало обучения
---------------
Внутри проекта необходимо создать папку, в которой будут храниться изображения и их аннотации.

```shell script
    mkdir your_folder
```

**Разметку можно сделать с помощью - https://github.com/tzutalin/labelImg**

Затем необходимо загрузить размеченные данные в your_folder.

1. Запустить функции из prepare_data.py для предобработки датасета

2. Перед запуском auto_object_detection.py можно выбрать свои параметры класса.

    * Модель можно выбрать из предложенных по следующей ссылке - https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md

Готовая модель
--------------

1. Файлы, сгенерированные кодом (pipeline.config, data.csv, label_map.pbtxt, train_data.record, test_data.record)

    * pipeline.config — содержит все параметры для обучения конкретной модели (/your_folder)
    
    * label_map.pbtxt — файл с описанием классов (/annotations)
    
    * data.csv — файл с данными об изображениях и их разметках (/your_folder)
    
    * train_data.record, test_data.record — файл с данными из data.csv в формате, необходимом tensorflow для обучения (/annotations)
    
2. Все чекпоинты модели находятся в папке /your_folder/output

3. Готовая модель находится в папке saved_model, your_folder/output/frozen/saved_model


