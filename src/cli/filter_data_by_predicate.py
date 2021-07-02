import pandas as pd
import click
from log import get_logger
from utils import load_triples_from_tsv

logger = get_logger(__name__)


@click.command()
@click.option("-i", "--input_path", required=True)
@click.option("-o", "--output_path", required=True)
@click.option("-p", "--predicate_filter_path", required=True)
def filter_triples_tsv_by_predicates(
    input_path: str, output_path: str, predicate_filter_path: str
):
    triples = load_triples_from_tsv(input_path)
    predicate_filter = pd.read_csv(predicate_filter_path)
    predicates_keep = predicate_filter.loc[
        predicate_filter["keep"] == 1, "predicate"
    ].tolist()

    filtered_triples = triples[triples["predicate"].isin(predicates_keep)]
    filtered_triples.to_csv(output_path, sep="\t", index=False, header=False)
    logger.debug(
        f"Input no triples: {len(triples):,}. No predicates kept: {len(predicates_keep):,}/{len(predicate_filter):,}. Output no triples: {len(filtered_triples):,}."
    )


if __name__ == "__main__":
    filter_triples_tsv_by_predicates()
