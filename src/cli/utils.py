import time
import pandas as pd
from pykeen.triples import TriplesFactory


def get_timestamp():
    return time.strftime("%Y%m%d-%H%M")


def load_triples_from_tsv(input_path: str) -> pd.DataFrame:
    """
    Due to newline characters in triples, they can't be imported correctly by a simple `pd.read_csv` or `TriplesFactory.from_file()`.
    This function imports triples safely using pandas.
    """

    triples = pd.read_csv(
        input_path,
        names=["subject", "predicate", "object"],
        sep="\t",
        lineterminator="\n",
        dtype=str,
        na_filter=False,
    )

    triples["object"] = triples["object"].apply(
        lambda x: x[0:-1] if x.endswith("\r") else x
    )

    return triples


def triplesfactory_from_tsv(input_path, **triplesfactory_kwargs) -> TriplesFactory:
    """
    Create a `pykeen.triples.TriplesFactory` from a TSV. More reliable than `TriplesFactory.from_file` for our data.
    """

    triples = load_triples_from_tsv(input_path).astype(str).values

    return TriplesFactory.from_labeled_triples(triples, **triplesfactory_kwargs)
