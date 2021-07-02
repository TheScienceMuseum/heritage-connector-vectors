import time
import pandas as pd


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
