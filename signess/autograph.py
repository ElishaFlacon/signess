class ProcessingAutograph():
    """

    Класс для получения подписей из документов

    Основан на библиотеке inskrib

    """

    def __init__(self, autograph, document) -> None:
        self.__autograph = autograph
        self.__document = document

    def __processing(self, path: str) -> None:
        """
        Метод для получения подписей из документов
            - path - путь сохранения обработанных подписей
        """
        self.__document.get_authoraphs(path, self.__autograph)
