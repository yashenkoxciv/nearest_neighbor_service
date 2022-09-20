import uuid
import numpy as np
from vnnm.collection import Collection


def gen_examples(n_examples, dims, save_path):
    idxs = [uuid.uuid4().hex for _ in range(n_examples)]

    vectors = np.random.randn(n_examples, dims)

    c = Collection(idxs, vectors)
    c.save_pickle(save_path)


if __name__ == '__main__':
    config = [
        (10, 150, 'tests/data/collection_small.pkl'),
        (10000, 2048, 'tests/data/collection_test_work_views.pkl'),
        (90, 2048, 'tests/data/collection_temporal_views.pkl'),
        (49, 2048, 'tests/data/collection_cards.pkl'),
    ]

    for n, d, pkl_path in config:
        print(pkl_path, flush=True)
        gen_examples(n, d, pkl_path)

