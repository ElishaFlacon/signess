import os
import numpy as np

from signess.dataset import Dataset
from signess.network import FedotCNN

from inskrib.autograph import Autograph
from inskrib.documents import Document

from fedot.core.data.data import InputData, OutputData


def create_dataset():
    path_to_data = "./example/docs"

    dataset = Dataset(Autograph(), Document())
    path_to_dataset = dataset.generate(path_to_data)

    return path_to_dataset


def test_network():
    path_to_save_and_load = './model'
    path_to_picture = './result/autographs/0-first_person-0.png'

    network = FedotCNN()

    assert network
    assert network._FedotCNN__model.is_fitted == False

    path_to_dataset = create_dataset()
    datasetIsExist = os.path.exists(path_to_dataset)

    assert path_to_dataset
    assert datasetIsExist == True

    dataset = network.load_dataset(path_to_dataset)

    assert dataset
    assert type(dataset) == InputData

    network.train(dataset, 3)

    assert network
    assert network._FedotCNN__model.is_fitted == True

    predicts = network.predict(dataset)

    assert predicts
    assert type(predicts) == OutputData

    # ERROR from Fedot
    # classify = network.classify(path_to_picture, path_to_dataset)
    # assert classify
    # assert type(classify) == np.ndarray
    # assert classify[0] >= 0.8

    network.save(path_to_save_and_load)
    modelIsExist = os.path.exists(path_to_save_and_load)

    assert modelIsExist == True

    # IF FEDOT < 0.7.2 version THIS DO NOT WORK

    # new_network = FedotCNN()

    # assert new_network
    # assert new_network._FedotCNN__model.is_fitted == False

    # new_network.load(path_to_save_and_load)

    # assert new_network
    # assert new_network._FedotCNN__model.is_fitted == True

    # new_dataset = new_network.load_dataset(path_to_dataset)

    # assert new_dataset
    # assert type(new_dataset) == InputData

    # new_predicts = new_network.predict(new_dataset)

    # assert new_predicts
    # assert type(new_predicts) == OutputData

    # new_classify = new_network.classify(path_to_picture, path_to_dataset)

    # assert new_classify
    # assert type(new_classify) == np.ndarray
    # assert new_classify[0] >= 0.8

    # check? this can do not work!
    # assert classify == new_classify[0]
