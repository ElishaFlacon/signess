from signess.dataset import Dataset
from signess.networks.reepsy import ReepsyCNN

from inskrib.autograph import Autograph
from inskrib.documents import Document


# конфиг
path_to_docs = './docs'
path_to_autographs = './dataset/autographs'
path_to_csv = './dataset/filenames.csv'
path_to_picture = './example/picture.png'

# создали инстанс класса датасет, в котором указали какой датасет нам нужен
dataset = Dataset(
    autograph=Autograph(size=(360, 360)),
    document=Document(),
    base="reepsy"
)

# генерация датасета из директории с документами
dataset.generate(path=path_to_docs)

# создали инстанс необученной нейронной сети
network = ReepsyCNN()

# полный пайплайн обучения модели
(model, accuracy, dataset) = network.process(
    data=path_to_autographs,
    csv=path_to_csv,
    num_epochs=3
)

# классифицируем изображение
predict = network.classify(model, path_to_picture)
print(f'Predict: {predict}')
