from signess.networks.base import BaseCNN

from torch.utils.data import DataLoader

from reepsy.core.network import Network
from reepsy.core.data import Data
from reepsy.core.classification import Classification


class ReepsyCNN(BaseCNN):
    """

    Класс для создания нейронной сети на основе библиотеки Reepsy

    Базовое описание методов находиться в base.py

    При инициализации создается необученная нейронная сеть от Reepsy

    """

    def __init__(self) -> BaseCNN:
        self.__network = Network()

    def load_dataset(self, data: str, csv: str) -> DataLoader:
        """
        Метод для загрузки датасета:
            - data - путь к директории с изображениями
            - csv - путь к csv файлу

        Возвращает объект DataLoader (из PyTorch)
        """
        dataset = Data.load_dataset(
            dataset_data_path=data,
            dataset_csv_path=csv,
        )
        return dataset

    def train(self, dataset: str, num_epochs: int = 1) -> any:
        """
        Метод для обучения модели:
            - dataset - объект DataLoader (из PyTorch)
            - num_epochs - количество эпох, по стандарту = 1

        Возвращает обученную модуль
        """
        model = self.__network.train(dataset=dataset, num_epochs=num_epochs)
        return model

    def accuracy(self, dataset: str) -> float:
        """
        Метод для проверки точности модели:
            -  dataset - объект DataLoader (из PyTorch)

        Возвращает float число, которое говорит о проценте правильных ответов 
        """
        accuracy = self.__network.predict(dataset=dataset)
        return accuracy

    def process(self, data: str, csv: str, num_epochs: int = 1):
        """
        Метод для автоматического выполнения пайплайна:
            - data - путь к директории с изображениями
            - csv - путь к csv файлу
            - num_epochs - количество эпох, по стандарту = 1

        Возвращает (обученную модель, процент правильных ответов, объект DataLoader (из PyTorch))
        """
        dataset = self.load_dataset(data, csv)
        model = self.train(dataset, num_epochs)
        accuracy = self.accuracy(dataset)
        return (model, accuracy, dataset)

    def classify(self, model: any, picture_path: str) -> int:
        """
        Метод классификации изображения
            - picture_path - путь к изображению

        Возвращает id объекта
        """
        result = Classification.classify_picture(
            model=model,
            picture_path=picture_path
        )
        return result
