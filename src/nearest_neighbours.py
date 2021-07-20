"""
Submodule for finding nearest neighbours.
"""

from typing import Iterable, Tuple
import faiss
from src.embedding_store import KGEmbeddingStore


class FaissNearestNeighbours:
    def __init__(self, embedding_store: KGEmbeddingStore):
        """Create a wrapper around a Faiss IndexFlatL2, which gets nearest neighbours based on Euclidean distance

        Args:
            embedding_store (KGEmbeddingStore): initialised `embedding_store.KGEmbeddingStore`
        """
        self.embedding_store = embedding_store
        self.faiss_index = None

    def fit(self, entities_or_relations: str) -> "FaissNearestNeighbours":
        """Fit Faiss index using either entity or relation data.

        Args:
            entities_or_relations (str): whether train the Faiss index to retrieve entities ("entities") or relations ("relations")
        """
        if entities_or_relations == "entities":
            X = self.embedding_store.ent_embedding_matrix
        elif entities_or_relations == "relations":
            X = self.embedding_store.rel_embedding_matrix
        else:
            raise ValueError(
                "Argument `entities_or_relations` must be either 'entities' or 'relations'."
            )

        self.faiss_index = faiss.IndexFlatL2(X.shape[1])
        self.faiss_index.add(X)

        return self

    def search(self, entities: Iterable[str], k: int) -> Tuple[list]:
        """Get `k` nearest neighbours for each entity in `entities`.

        Args:
            entities (Iterable[str]): collection of entities by their URI/value
            k (int): number of nearest neighbours to return for each entity

        Returns:
            Tuple[list]: entities, distances. Each is a list of lists where the order corresponds to the order of entities in the function call
        """
        xq = self.embedding_store.get_entity_embeddings(entities)
        distances, idxs = self.faiss_index.search(xq, k)
        entities = [self.embedding_store.idxs_to_entities(_) for _ in idxs]

        # we reverse here because Faiss returns the values in the reverse order
        return entities[::-1], distances[::-1]
