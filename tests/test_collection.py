import pickle
import numpy as np
from vnnm.collection import Collection
from sklearn.neighbors import NearestNeighbors


def collection_processing(test_collection):
    test_collection = 'tests/data/collection_small.pkl'

    col = Collection.load_from_pickle('tests/data/collection_small.pkl')

    # native NearestNeighbors
    with open(test_collection, 'rb') as f:
        idxs, vectors = pickle.loads(f.read())
    
    nn = NearestNeighbors(
        n_neighbors=3,
        algorithm='brute',
        metric='minkowski'
    )
    nn.fit(vectors)

    # test
    cnns = col.find_nearest_neighbors(vectors[:1])
    
    nns = nn.kneighbors(vectors[:1], 2, return_distance=False)

    return cnns, nns


def test_collection_small():
    cnns, nns = collection_processing('tests/data/collection_small.pkl')

    assert np.allclose(cnns, nns)


def test_collection_cards():
    cnns, nns = collection_processing('tests/data/collection_cards.pkl')

    assert np.allclose(cnns, nns)




