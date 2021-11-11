import argparse
import os
from typing import List
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
import uvicorn
from src.embedding_store import KGEmbeddingStore
from src.nearest_neighbours import FaissNearestNeighbours
from pathlib import Path
from dotenv import load_dotenv
from src.cli.log import get_logger

logger = get_logger(__name__)

load_dotenv()

EMBEDDINGS_FILE_NAMES = [
    Path(os.environ.get("ENTITY_EMBEDDING_PATH")).name,
    Path(os.environ.get("RELATION_EMBEDDING_PATH")).name,
]

assert (
    Path(os.environ.get("ENTITY_EMBEDDING_PATH")).parent
    == Path(os.environ.get("RELATION_EMBEDDING_PATH")).parent  # noqa: W503
)

EMBEDDINGS_FOLDER = os.path.join(
    os.path.dirname(__file__),
    "..",
    Path(os.environ.get("ENTITY_EMBEDDING_PATH")).parent,
)

MAPPINGS_FILE_NAMES = [
    Path(os.environ.get("ENTITY_MAPPING_PATH")).name,
    Path(os.environ.get("RELATION_MAPPING_PATH")).name,
]

assert (
    Path(os.environ.get("ENTITY_MAPPING_PATH")).parent
    == Path(os.environ.get("RELATION_MAPPING_PATH")).parent  # noqa: W503
)

MAPPINGS_FOLDER = os.path.join(
    os.path.dirname(__file__), "..", Path(os.environ.get("ENTITY_MAPPING_PATH")).parent
)

embedding_store = KGEmbeddingStore.from_dglke(
    embeddings_folder=EMBEDDINGS_FOLDER,
    embeddings_file_names=EMBEDDINGS_FILE_NAMES,
    mappings_folder=MAPPINGS_FOLDER,
    mappings_file_names=MAPPINGS_FILE_NAMES,
)
faiss_index = FaissNearestNeighbours(embedding_store).fit("entities")

app = FastAPI()


class NeighboursRequest(BaseModel):
    entities: List[str]
    k: int


@app.post("/neighbours")
async def get_nearest_neighbours(request: NeighboursRequest):
    neighbours, distances = faiss_index.search(request.entities, request.k + 1)

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


class DistanceRequest(BaseModel):
    entity_a: str
    entity_b: str


@app.post("/distance")
async def get_distance(request: DistanceRequest):
    logger.debug(f"DISTANCES: ent_a {request.entity_a}, ent_b {request.entity_b}")

    if request.entity_a == request.entity_b:
        return 0

    return embedding_store.get_entity_distance([request.entity_a, request.entity_b])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--port", type=int, help="Optional port (default 8000)", default=8000
    )

    args = parser.parse_args()
    port = args.port

    uvicorn.run(app, host="0.0.0.0", port=port)
