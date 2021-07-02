import click
import numpy as np
import pandas as pd
from log import get_logger
from utils import load_triples_from_tsv

logger = get_logger(__name__)


@click.command()
@click.option("-i", "--input_path", required=True)
@click.option("-o", "--output_path", required=True)
@click.option("-k", "--keep_subjects_proportion", type=float, required=True)
@click.option("-r", "--random_state", type=int, default=100)
def make_smaller_triples(
    input_path: str,
    output_path: str,
    keep_subjects_proportion: float,
    random_state: int,
):
    """
    Make smaller triples file for testing, by keeping triples only from a subset of subjects (i.e. SMG records).
    The value of `keep_subjects_proportion` must be under 1, e.g. .3 keeps 30%.
    """

    triples = load_triples_from_tsv(input_path)

    # Split the triples into those with subject from SMG (main graph), and those with subject from Wikidata
    # (Wikidata cache).
    smg_triples = triples[triples["subject"].str.contains("sciencemuseum")]
    wikidata_triples = triples[triples["subject"].str.contains("wikidata")]

    assert len(wikidata_triples) + len(smg_triples) == len(triples)

    # Get a random sample of subjects from SMG
    unique_subjects = smg_triples["subject"].unique()
    rnd = np.random.RandomState(random_state)
    unique_subjects_small = rnd.choice(
        unique_subjects, int(keep_subjects_proportion * len(unique_subjects))
    )

    # Get all the triples from the graph which either have one of the selected SMG entities as subject or object.
    smg_triples_small = triples[
        triples["subject"].isin(unique_subjects_small)
        | triples["object"].isin(unique_subjects_small)  # noqa: W503
    ]

    # Join this with the relevant part of the Wikidata cache: the triples whose subject is an object in the above set.
    relevant_wikidata_cache = wikidata_triples[
        wikidata_triples["subject"].isin(smg_triples_small["object"].unique().tolist())
    ]
    smaller_triples = pd.concat(
        [smg_triples_small, relevant_wikidata_cache], axis=0, ignore_index=True
    )

    smaller_triples.to_csv(output_path, sep="\t", index=False, header=False)

    # Calculate and display some stats
    stats_triples_perc = round(len(smaller_triples) / len(triples) * 100, 2)
    stats_entities_perc = round(
        len(unique_subjects_small) / len(unique_subjects) * 100, 2
    )

    logger.debug(
        f"{len(smaller_triples):,} ({stats_triples_perc}%) triples for {len(unique_subjects_small):,} ({stats_entities_perc}%) entities saved to {output_path}."
    )


if __name__ == "__main__":
    make_smaller_triples()
