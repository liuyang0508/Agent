from enum import Enum
import faiss
import numpy as np

class IndexType(Enum):
    L2 = (lambda dim: faiss.IndexFlatL2(dim), False)
    IP = (lambda dim: faiss.IndexFlatIP(dim), False)
    HNSWFlat = (lambda dim: faiss.IndexHNSWFlat(dim), True)
    IVFFlat = (lambda dim: faiss.IndexIVFFlat(dim), True)
    LSH = (lambda dim: faiss.IndexLSH(dim), False)
    ScalarQuantizer = (lambda dim: faiss.IndexScalarQuantizer(dim), True)
    PQ = (lambda dim: faiss.IndexPQ(dim), True)
    IVFScalarQuantizer = (lambda dim: faiss.IndexIVFScalarQuantizer(dim), True)
    IVFPQ = (lambda dim: faiss.IndexIVFPQ(dim), True)
    IVFPQR = (lambda dim: faiss.IndexIVFPQR(dim), True)

    def __init__(self, index_constructor, requires_training):
        self.index_constructor = index_constructor
        self.requires_training = requires_training

    @classmethod
    def from_string(cls, metric: str):
        try:
            return cls[metric.upper()]
        except KeyError:
            raise ValueError(f"Unsupported metric: {metric}")

    def build_index(self, dimension):
        return self.index_constructor(dimension)
    
class VectorIndexManager:
    def __init__(self, dimension: int, metric: str = 'L2'):
        self.dimension = dimension
        index_type = IndexType.from_string(metric)
        index_constructor = index_type.value.__getitem__(0)
        self.index = index_constructor(dimension)
        
    def initialize_index(self, training_vectors: np.ndarray, metric: str = 'L2'):
        """Initialize the index, which may include training if necessary."""
        index_type = IndexType.from_string(metric)
        
        if index_type.value.__getitem__(1) and training_vectors is not None:
            self.index.train(training_vectors)
        else:
            pass

        return self.index.is_trained
    
    def add_vectors(self, vectors: np.ndarray):
        """Add vectors to the index."""
        assert vectors.shape[1] == self.dimension
        self.index.add(vectors.astype('float32'))
        # Save index
        faiss.write_index(self.index, "index_file.index")
        return self.index.ntotal

    def search_vectors(self, queries: np.ndarray, k: int, axis: int = 0):
        """Search for the k nearest neighbors of each query vector."""
        # Load index
        self.index = faiss.read_index("index_file.index")
        if axis == 0:
            assert queries.shape[0] == self.dimension
            distances, indices = self.index.search(np.expand_dims(queries, axis=0).astype('float32'), k)
        else:
            assert queries.shape[1] == self.dimension
            distances, indices = self.index.search(queries.astype('float32'), k)
        return indices, distances

    def delete_vectors(self, vector_ids: list):
        """Delete vectors from the index (conceptual implementation)."""
        raise NotImplementedError("faiss does not directly support deleting vectors. "
                                  "This method is provided as a placeholder for a hypothetical implementation.")

    def update_index_structure(self, new_metric: str):
        """Update the index structure with a new distance metric."""
        self.index_type = IndexType.from_string(new_metric)
        self.index = self.index_type.build_index(self.dimension)        

        return self.index_type.requires_training
    