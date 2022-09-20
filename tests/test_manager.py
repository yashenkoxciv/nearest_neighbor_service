import pickle
import numpy as np
from vnnm.manager import Manager
from sklearn.neighbors import NearestNeighbors


test_collections = [
    'tests/data/collection_small.pkl',
    'tests/data/collection_cards.pkl'
]


def collection_processing(collection_name):
    man = Manager.load_from_pickle_files(test_collections)

    # native NearestNeighbors
    nns = {}
    for test_collection in test_collections:
        with open(test_collection, 'rb') as f:
            idxs, vectors = pickle.loads(f.read())
        
        nn = NearestNeighbors(
            n_neighbors=3,
            algorithm='brute',
            metric='minkowski'
        )
        nn.fit(vectors)
        

    # test
    cnns = man.get_nearest_neighbors(collection_name, vectors[:1])
    
    nns = nn.kneighbors(vectors[:1], 2, return_distance=False)

    return cnns, nns


def test_collection_small():
    cnns, nns = collection_processing('tests/data/collection_small.pkl')

    assert np.allclose(cnns, nns)


def test_collection_cards():
    cnns, nns = collection_processing('tests/data/collection_cards.pkl')

    assert np.allclose(cnns, nns)

