import argparse
import os
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
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


class NeighboursRequest(BaseModel):
    entities: List[str]
    k: int


@app.post("/neighbours")
async def get_nearest_neighbours(request: NeighboursRequest):
    neighbours, distances = faiss_index.search(request.entities, request.k)

    response = {}

    for idx, ent in enumerate(request.entities):
        if ent != neighbours[idx][0]:
            raise HTTPException(
                status_code=404,
                detail=f"It looks like there's a mismatch between a request entity and its nearest neighbour. Problem entity: {ent}",
            )

        response.update(
            {ent: list(zip(neighbours[idx][1:], distances[idx][1:].tolist()))}
        )

    return response


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int, help="Optional port (default 8000)", default=8000
    )

    args = parser.parse_args()
    port = args.port

    uvicorn.run(app, host="0.0.0.0", port=port)
