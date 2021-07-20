import os
from typing import Iterable, List
import numpy as np
import pandas as pd


class KGEmbeddingStore:
    """Provides a consistent interface to access KG embeddings."""

    def __init__(
        self,
        ent_embeddings: np.ndarray,
        ent_mapping: pd.DataFrame,
        rel_embeddings: np.ndarray,
        rel_mapping: pd.DataFrame,
    ):
        """Create KGEmbeddingStore from embeddings matrices and mappings.
        Embeddings matrices are np.ndarrays with dtype=float32; mappings are DataFrames where the index column corresponds to rows of the embeddings matrices,
        and there is a 'value' column with a unique identifier for each entity or relation.

        Args:
            ent_embeddings (np.ndarray)
            ent_to_idx (pd.DataFrame)
            rel_embeddings (np.ndarray)
            rel_to_idx (pd.DataFrame)
        """
        self._ent_embeddings = ent_embeddings
        self._rel_embeddings = rel_embeddings
        self._ent_mapping = ent_mapping
        self._rel_mapping = rel_mapping

    @property
    def ent_embedding_matrix(self):
        return self._ent_embeddings

    @property
    def rel_embedding_matrix(self):
        return self._rel_embeddings

    @property
    def entity_dim(self):
        return self._ent_embeddings.shape[1]

    @property
    def relation_dim(self):
        return self._rel_embeddings.shape[1]

    def get_entity_embeddings(self, entities: Iterable[str] = None) -> np.ndarray:
        """Get embeddings vectors for a subset of entities.

        Args:
            entities (Iterable[str], optional): List of entities by URI/value. Defaults to None.

        Returns:
            np.ndarray: rows correspond to order of `entities`
        """
        if not entities:
            return self.ent_embedding_matrix

        tempdf = self._ent_mapping[self._ent_mapping["value"].isin(entities)]
        # Here we create a new column which allows us to ensure the returned idxs match up to the order
        # of `entities`.
        tempdf["sort_entities"] = pd.Categorical(
            tempdf["value"], categories=entities, ordered=True
        )
        idxs = tempdf.sort_values("sort_entities").index.tolist()

        return self._ent_embeddings[idxs, :]

    def get_relation_embeddings(self, relations: Iterable[str] = None) -> np.ndarray:
        """Get embeddings vectors for a subset of relations.

        Args:
            relations (Iterable[str], optional): List of relations by URI/value. Defaults to None.

        Returns:
            np.ndarray: rows correspond to order of `relations`
        """

        if not relations:
            return self.rel_embedding_matrix

        idxs = self._rel_mapping[
            self._rel_mapping["value"].isin(relations)
        ].index.tolist()

        return self._rel_embeddings[idxs, :]

    def idxs_to_entities(self, idxs: int) -> List[str]:
        """Convert indexes to entity values."""
        return self._ent_mapping.iloc[idxs]["value"].tolist()

    def idxs_to_relations(self, idxs: int) -> List[str]:
        """Convert indexes to relation values."""
        return self._rel_embeddings.iloc[idxs]["value"].tolist()

    @classmethod
    def from_dglke(
        cls,
        embeddings_folder: str,
        embeddings_file_names: Iterable[str],
        mappings_folder: str,
        mappings_file_names: Iterable[str] = ["entities.tsv", "relations.tsv"],
    ) -> "KGEmbeddingStore":
        """Create a KGEmbeddingStore from saved DGL-KE training outputs.

        Args:
            embeddings_folder (str): folder that entity and relation embeddings matrices (.npy files) are stored in
            embeddings_file_names (Iterable[str]): file names of embedding and relation matrices in `embeddings_folder`
            mappings_folder (str): folder that the TSV files storing the mapping between matrix rows and entity/relation names are stored in
            mappings_file_names (Iterable[str], optional): file names of entity and relation mapping TSVs. Defaults to ["entities.tsv", "relations.tsv"].

        Returns:
            KGEmbeddingStore: correctly initialised instance
        """
        entities = np.load(
            os.path.join(embeddings_folder, embeddings_file_names[0])
        ).astype("float32")
        relations = np.load(
            os.path.join(embeddings_folder, embeddings_file_names[1])
        ).astype("float32")

        ent_mapping = pd.read_csv(
            os.path.join(mappings_folder, mappings_file_names[0]),
            sep="\t",
            index_col=0,
            header=None,
            names=["value"],
        )
        rel_mapping = pd.read_csv(
            os.path.join(mappings_folder, mappings_file_names[1]),
            sep="\t",
            index_col=0,
            header=None,
            names=["value"],
        )

        return KGEmbeddingStore(
            ent_embeddings=entities,
            ent_mapping=ent_mapping,
            rel_embeddings=relations,
            rel_mapping=rel_mapping,
        )
