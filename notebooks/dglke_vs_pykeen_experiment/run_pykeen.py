# flake8: noqa

MODEL = "RotatE"
N_EPOCHS = 100
EMBEDDING_DIM = 400
BATCH_SIZE = 2000
LR = 0.02
NUM_NEGS_PER_POS = 50
BATCH_SIZE_EVAL = 8

DATA_FOLDER = "../../data/interim/train_test_split/"

from pykeen.pipeline import pipeline
import sys

sys.path.append("../..")

from src.cli.utils import triplesfactory_from_tsv

train = triplesfactory_from_tsv(DATA_FOLDER + "train.csv")
test = triplesfactory_from_tsv(DATA_FOLDER + "test.csv")
val = triplesfactory_from_tsv(DATA_FOLDER + "val.csv")

result = pipeline(
    training=train,
    testing=test,
    validation=val,
    model=MODEL,
    model_kwargs=dict(embedding_dim=EMBEDDING_DIM),
    optimizer_kwargs=dict(lr=LR),
    training_kwargs=dict(
        num_epochs=N_EPOCHS,
        batch_size=BATCH_SIZE,
    ),
    negative_sampler_kwargs=dict(
        num_negs_per_pos=NUM_NEGS_PER_POS,
    ),
    evaluation_kwargs=dict(
        batch_size=BATCH_SIZE_EVAL,
    ),
    evaluator_kwargs=dict(filtered=True),
)
