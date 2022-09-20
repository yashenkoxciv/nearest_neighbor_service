import os
import logging
from collection import Collection


class CollectionManager:
    """ {collection name -> Collection}
    """
    def __init__(self, named_collections=None):
        if named_collections is None:
            named_collections = {}
        
        self.named_collections = named_collections
    
    def load_from_pickle_file(pkl_path):
        collection_name = os.path.split(pkl_path)[-1].split('.')[0]
        collection = Collection.load_from_pickle(pkl_path)
        
        if collection_name in self.named_collections:
            logging.warning(f'{collection_name} collection is overwritten')
    
        self.named_collections[collection_name] = collection
    
    def load_from_pickle_files(pkl_paths):
        for pkl_path in pkl_paths:
            self.load_from_pickle_file(pkl_path)
    
    def remove_collection(self, collection_name):
        del self.named_collections[collection_name]
    
    def get_collection(self, collection_name):
        return self.named_collections[collection_name]

    def get_neighbors(self, collection_name, query_vectors):
        collection = self.named_collections[collection_name]
        nns = collection.find_nearest_neighbors(query_vectors)
        return nns


