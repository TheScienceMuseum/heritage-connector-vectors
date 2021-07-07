import os
import click
import pandas as pd
from log import get_logger, DisableLogger
from utils import triplesfactory_from_tsv

logger = get_logger(__name__)


@click.command()
@click.option("-i", "--input_path", required=True)
@click.option("-o", "--output_path", required=True)
@click.option(
    "-s",
    "--sizes",
    type=str,
    required=True,
    help="Comma-separated list of train/test/val or train/test sizes. If just a float is passed, this is used as the train size, with the rest of the triples making up the test set.",
)
@click.option("-r", "--random_state", type=int, default=100)
def run_train_test_split(input_path, output_path, sizes, random_state):
    tf = triplesfactory_from_tsv(input_path)

    sizes = [float(i) for i in sizes.split(",")]

    if len(sizes) <= 2:
        train, test = tf.split(sizes, random_state=random_state)
        val = None
    elif len(sizes) == 3:
        train, test, val = tf.split(sizes, random_state=random_state)
    else:
        raise ValueError(
            "Incorrect number of parameters passed for option `--sizes`. See PyKEEN's `TriplesFactory.split()` documentation."
        )

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with DisableLogger():
        train_triples = train.triples
        test_triples = test.triples
    pd.DataFrame(train_triples).to_csv(
        os.path.join(output_path, "train.csv"), sep="\t", index=False, header=False
    )
    pd.DataFrame(test_triples).to_csv(
        os.path.join(output_path, "test.csv"), sep="\t", index=False, header=False
    )
    success_msg = f"Created split of {train_triples.shape[0]} train triples; {test_triples.shape[0]} test triples"
    if val:
        with DisableLogger():
            val_triples = val.triples
        pd.DataFrame(val_triples).to_csv(
            os.path.join(output_path, "val.csv"), sep="\t", index=False, header=False
        )
        success_msg += f"; {val_triples.shape[0]} validation triples"

    logger.info(success_msg)


if __name__ == "__main__":
    run_train_test_split()
