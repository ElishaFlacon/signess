from signess.autograph import ProcessingAutograph


class Dataset(ProcessingAutograph):
    """

    Класс для автоматического сбора датасета под каждую нейронную сеть
        - autograph - инстанс класса inskrib.Autograph
        - document - инстанс класса inskrib.Document
        - base - для какой нейронной сети будет генерироваться датасет:
            - "reepsy" 
            - "fedot"

    Также наследует класс ProcessingAutograph

    """

    def __init__(self, autograph, document, base: str = 'reespy') -> None:
        super().__init__(self, autograph, document)
        self.__base = base

    def generate(self, path: str) -> str:
        """
        Метод для генерации датасета
            - path - путь для сохранения датасета

        Возвращает путь сохранения датасета
        """
        if (self.__base == 'reepsy'):
            self.__document.set_grouping(False)

        if (self.__base == 'fedot'):
            self.__document.set_grouping(True)

        self.__processing(path, self.__autograph)
        return path
