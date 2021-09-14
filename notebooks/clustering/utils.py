import os
import json
import requests
import sys

sys.path.append("../..")

from src.embedding_store import KGEmbeddingStore  # noqa: E402

API_ENDPOINT = "https://d0rgkq.deta.dev/labels"


def get_labels(entities):
    payload = json.dumps({"uris": entities})
    headers = {"Content-Type": "application/json"}

    return requests.post(API_ENDPOINT, headers=headers, data=payload).json()


def load_embedding_store():
    MODEL_FOLDER = os.path.join(
        os.path.dirname(__file__), "../../data/processed/final_model_dglke"
    )
    embedding_store = KGEmbeddingStore.from_dglke(
        embeddings_folder=MODEL_FOLDER,
        embeddings_file_names=[
            "heritageconnector_RotatE_entity.npy",
            "heritageconnector_RotatE_relation.npy",
        ],
        mappings_folder=MODEL_FOLDER,
        mappings_file_names=["entities.tsv", "relations.tsv"],
    )

    return embedding_store
