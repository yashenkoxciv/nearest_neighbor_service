import pickle
from sklearn.neighbors import NearestNeighbors



class Collection:
    def __init__(self, idxs, vectors):
        self.idxs = idxs
        self.vectors = vectors

        self.nn = Collection._train_nearest_neighbor(vectors)
    
    @staticmethod
    def load_from_pickle(pkl_path):
        with open(pkl_path, 'rb') as f:
            col = pickle.loads(f.read())
        
        idxs, vectors = col
        
        return Collection(idxs, vectors)
    
    @staticmethod
    def _train_nearest_neighbor(vectors):
        nn = NearestNeighbors(
            n_neighbors=3,
            algorithm='brute',
            metric='minkowski'
        )
        nn.fit(vectors)
        return nn
    
    def find_nearest_neighbors(self, query_vectors):
        nearest_neighbors = self.nn.kneighbors(query_vectors, 2, return_distance=False)
        return nearest_neighbors
    
    def remove_vector(self, vector_idx):
        # remove vector from self.idxs and self.vectors
        vector_pos = self.idxs.index(vector_idx)
        del self.idxs[vector_pos]
        np.delete(self.vectors, vector_pos, axis=0)

        # train NearestNeighbors again
        self.nn = Collection._train_nearest_neighbor(vectors)

    def add_vector(self, idx, vector):
        # add vector to self.idxs and self.vectors
        self.idxs.append(idx)
        np.append(self.vectors, [vector], axis=0)
        
        # train NearestNeighbors again
        self.nn = Collection._train_nearest_neighbor(vectors)
    
    def save_pickle(self, pkl_path):
        with open(pkl_path, 'wb') as f:
            f.write(pickle.dumps((self.idxs, self.vectors)))
