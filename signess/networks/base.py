class BaseCNN():
    """

    Интерфейс для классов нейронных сетей

    При инициализации создается необученная нейронная сеть

    """

    def __init__(self):
        pass

    def load_dataset(self):
        """
        Метод для загрузки датасета
        """
        pass

    def train(self):
        """
        Метод для обучения модели
        """
        pass

    def accuracy(self):
        """
        Метод для проверки точности модели
        """
        pass

    def save(self):
        """
        Метод для сохранения модели
        """
        pass

    def load(self):
        """
        Метод для загрузки модели
        """
        pass

    def process(self):
        """
        Метод для автоматического выполнения пайплайна:
            load_dataset
            train
            predicts
        """
        pass

    def classify(self):
        """
        Метод классификации изображения
        """
        pass
