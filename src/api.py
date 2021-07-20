import argparse
import os
from fastapi import FastAPI
import uvicorn
from src.embedding_store import KGEmbeddingStore
from src.nearest_neighbours import FaissNearestNeighbours

MODEL_FOLDER = os.path.join(
    os.path.dirname(__file__), "../data/processed/final_model_dglke"
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
faiss_index = FaissNearestNeighbours(embedding_store).fit("entities")

app = FastAPI()


@app.get("/nearest_neighbours")
@app.post("/nearest_neighbours")
async def get_nearest_neighbours(entity: str, k: int):
    entities, distances = faiss_index.search([entity], k)
    entities = entities[0]
    distances = distances[0].tolist()

    return [(entities[idx], distances[idx]) for idx in range(len(entities))]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int, help="Optional port (default 8000)", default=8000
    )

    args = parser.parse_args()
    port = args.port

    uvicorn.run(app, host="0.0.0.0", port=port)
