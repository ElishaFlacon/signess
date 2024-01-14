from signess.dataset import Dataset
from signess.network import FedotCNN

from inskrib.autograph import Autograph
from inskrib.documents import Document

from fedot.core.utils import set_random_seed


def create_dataset(path_to_data):
    # create Dataset class
    ds = Dataset(
        autograph=Autograph(size=(380, 380)),
        document=Document()
    )

    # generate dataset by .npz file
    path_to_dataset = ds.generate(path=path_to_data)
    return path_to_dataset


def create_network(path_to_dataset, path_to_picture, path_to_save):
    # create Network
    network = FedotCNN()
    # load dataset where path - path to .npz file
    dataset = network.load_dataset(path=path_to_dataset)
    # train model with loaded dataset by 3 epochs
    network.train(dataset=dataset, num_epochs=3)

    # predicts dataset
    predicts = network.predict(dataset)
    print(f'Predicts: {predicts}')

    # classify picture
    predict = network.classify(path_to_picture, path_to_dataset)

    # save model
    network.save(path_to_save)
    print('Model Save!')

    return (predicts, predict)


def load_network(path_to_dataset, path_to_picture, path_to_load):
    # create new Network
    new_network = FedotCNN()
    # load model
    new_network.load(path_to_load)
    # load dataset where path - path to .npz file
    dataset = new_network.load_dataset(path=path_to_dataset)

    # predicts dataset on new model
    predicts = new_network.predict(dataset)
    print(f'Predicts: {predicts}')

    # classify picture on new model
    predict = new_network.classify(path_to_picture, path_to_dataset)
    print(f'Classify: {predict}')

    return (predicts, predict)


if __name__ == '__main__':
    set_random_seed(1)

    # config
    path_to_data = './example/datasets/mini'
    path_to_picture = './result/autographs/1-first_person-0.png'
    path_to_save_and_load = './model'

    path_to_dataset = create_dataset(path_to_data)
    create_network(path_to_dataset, path_to_picture, path_to_save_and_load)
    load_network(path_to_dataset, path_to_picture, path_to_save_and_load)
