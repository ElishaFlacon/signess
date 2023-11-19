import cv2
import numpy as np

from fedot.core.repository.tasks import Task, TaskTypesEnum
from fedot.core.data.data import InputData, OutputData
from fedot.core.pipelines.pipeline import Pipeline
from fedot.core.pipelines.node import PipelineNode


class FedotCNN():
    """

    Класс для создания нейронной сети на основе фреймворка Fedot

    При инициализации создается CNN на базе Fedot

    """

    def __init__(self):
        self.__model = self.__pipeline(True)
        self.__task = Task(TaskTypesEnum.classification)

    def __pipeline(self, composite_flag: bool = True) -> Pipeline:
        """
        Приватный метод для создания структуры нейронной сети, принимает:
            - composite_flag -

        Возвращает Pipeline из Fedot
        """

        node_first = PipelineNode('cnn')
        node_first.parameters = {
            'architecture': 'deep',
            'epochs': 15,
            'batch_size': 128
        }
        node_second = PipelineNode('cnn')
        node_second.parameters = {
            'architecture_type': 'simplified',
            'epochs': 10,
            'batch_size': 128
        }
        node_final = PipelineNode('rf', nodes_from=[node_first, node_second])

        if not composite_flag:
            node_final = PipelineNode('rf', nodes_from=[node_first])

        pipeline = Pipeline(node_final)
        return pipeline

    def __check_train(self):
        if (not self.__model.is_fitted):
            raise ValueError('ERROR: Model is not Train!')

    def load_dataset(self, path: str) -> InputData:
        """
        Метод для загрузки датасета, принимает:
            - path - путь до датасет файла .npz 

        Возвращает InputData из Fedot
        """

        with np.load(path) as data:
            x_train, y_train = data['DataX'], data['DataY']
            x_train = x_train[..., np.newaxis]

            ready_dataset = InputData.from_image(
                images=x_train,
                labels=y_train,
                task=self.__task
            )

            return ready_dataset

    def train(self, dataset: InputData, num_epochs: int) -> None:
        """
        Метод для тренировки модели, принимает:
            - dataset - датасет в виде InputData, который можно получить из метода load_dataset
            - num_epochs - количество эпох для обучения модели

        Возвращает None
        """

        self.__model.fit(input_data=dataset, n_jobs=num_epochs)

    def blunt(self) -> None:
        """
        Метод для сброса обучения у модели

        Возвращает None
        """

        self.__model.unfit()

    def accuracy(self, dataset: InputData) -> OutputData:
        """
        Метод для проверки точности модели, принимает:
            - dataset - датасет в виде InputData, который можно получить из метода load_dataset

        Возвращает OutputData из Fedot, в котором находятся данные о точности модели
        """

        self.__check_train()

        predictions = self.__model.predict(dataset)
        return predictions

    def classify(self, path_to_picture: str, path_to_dataset: str) -> np.ndarray:
        """
        Метод для классификации изображения, принимает:
            - path_to_picture - путь до изображения, которое нужно классифицировать
            - path_to_dataset - путь до датасет файла .npz (это нужно для получения количества классов)

        Возвращает ndarray из numpy, в котором находятся данные о классификации
        """

        self.__check_train()

        with np.load(path_to_dataset) as data:
            y_train = data['DataY']
            y_train = y_train[..., np.newaxis]

        picture = cv2.imread(path_to_picture)
        picture = cv2.resize(picture, (380, 380), 3)
        picture = np.reshape(picture, (1, 380, 380, 3))

        ready_picture = InputData.from_image(
            images=picture,
            labels=y_train,
            task=self.__task
        )

        predict = self.__model.predict(ready_picture)
        return predict.predict

    def save(self, path: str) -> None:
        """
        Метод для сохранения модели, принимает:
            - path - путь для сохранения модели

        Возвращает None
        """

        self.__model.save(path=path, create_subdir=False)

    def load(self, path: str) -> None:
        """
        Метод для загрузки модели, принимает:
            - path - путь до сохраненной модели

        Возвращает None
        """

        self.__model = Pipeline().load(path)
