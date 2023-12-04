import os
from signess.dataset import Dataset
from inskrib.autograph import Autograph
from inskrib.documents import Document


def test_dataset():
    path_to_data = './example/docs'
    path_to_save = "./result/dataset.npz"

    autograph = Autograph()
    document = Document()

    assert autograph
    assert document

    dataset = Dataset(autograph, document)

    assert dataset
    assert dataset._Dataset__autograph == autograph
    assert dataset._Dataset__document == document

    path_to_dataset = dataset.generate(path_to_data)

    assert path_to_dataset == path_to_save

    isExist = os.path.exists(path_to_dataset)

    assert isExist == True
