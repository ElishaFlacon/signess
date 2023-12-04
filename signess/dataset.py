import os
import cv2
import numpy as np

from inskrib.autograph import Autograph
from inskrib.documents import Document


class Dataset():
    """

    Класс для автоматического сбора датасета
        - autograph - инстанс класса inskrib.Autograph
        - document - инстанс класса inskrib.Document

    """

    def __init__(self, autograph: Autograph, document: Document) -> None:
        self.__document = document
        self.__autograph = autograph

    def package(self,  path_to_files: str, path_to_save: str) -> str:
        """
        Метод для сборки готового датасета в .npz формат
            - path_to_files - путь к директории с изображениями
            - path_to_save - путь сохранения .npz файла

        Возвращает путь сохранения датасета
        """
        vectorized_images_x = []
        vectorized_images_y = []

        for _, file in enumerate(os.listdir(path_to_files)):
            image = cv2.imread(path_to_files + file)
            image_array = np.array(image)
            vectorized_images_x.append(image_array)

            y = file.split('-')[0]
            vectorized_images_y.append(y)

        np.savez(
            path_to_save,
            DataX=vectorized_images_x,
            DataY=vectorized_images_y
        )

        return path_to_save

    def generate(self, path: str) -> str:
        """
        Метод для генерации датасета
            - path - путь для загрузки документов

        Возвращает путь сохранения датасета
        """
        path_to_files = "./result/autographs/"
        path_to_save = "./result/dataset.npz"

        self.__document.get_authoraphs(path, self.__autograph)
        path_to_save = self.package(path_to_files, path_to_save)

        return path_to_save
