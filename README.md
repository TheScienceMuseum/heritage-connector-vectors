heritage-connector-vectors
==============================

Generating graph and language embeddings for the Heritage Connector project.

## What is this?

A library that:
1. Takes JSON data of collection item descriptions or TSV/CSV data of triples, exported from [`heritage-connector`](https://github.com/TheScienceMuseum/heritage-connector).
2. Calculates sentence (using [`sentence-transformers`](https://github.com/UKPLab/sentence-transformers)) or graph (using [`pyKEEN`](https://github.com/pykeen/pykeen)) embeddings for this data.
3. Provides a unified interface for accessing these embeddings.

## Commands

* set up (install requirements and pre-commit hooks): `make init`
* run api for nearest neighbour search: `python -m src.api -p {port}`