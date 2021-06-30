import click
from pykeen.pipeline import pipeline_from_config
from pykeen.triples import TriplesFactory
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from utils import get_timestamp


@click.command()
@click.option(
    "-i",
    "--input",
    "input_data_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to input file (TSV of triples).",
)
@click.option(
    "-o",
    "--output",
    "output_path",
    type=click.Path(dir_okay=True, file_okay=False),
    required=True,
    help="Output folder to store the model output. Created if it does not already exist.",
)
@click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to config JSON.",
)
@click.option(
    "--gpu/--cpu",
    is_flag=True,
    default=False,
    help="Specify device to train on. Default is CPU; to train on GPU, pass `--gpu`.",
)
@click.option("--plot_losses/--no_plot_losses", is_flag=True, default=True)
@click.option("-s", "--random_seed", type=int, default=100)
def main(input_data_path, output_path, config_path, gpu, plot_losses, random_seed):

    # Load config and add data paths and device to the metadata that are stored with the model
    with open(config_path, "r") as f:
        config = json.load(f)

    config["metadata"].update(
        {
            "input_data_path": input_data_path,
            "config_path": config_path,
            "device": "gpu" if gpu else "cpu",
            "random_seed": random_seed,
            "datetime": get_timestamp(),
        }
    )

    # Run model and save to output directory
    tf = TriplesFactory.from_path(input_data_path, create_inverse_triples=False)
    pipeline_kwargs = dict(
        training=tf, testing=tf, device="gpu" if gpu else "cpu", random_seed=random_seed
    )

    results = pipeline_from_config(config, **pipeline_kwargs)
    results.save_to_directory(output_path)

    # Save embeddings matrices
    ent_representations = results.model.entity_representations
    rel_representations = results.model.relation_representations

    with open(os.path.join(output_path, "entity_embeddings.npy"), "wb") as f:
        np.save(f, ent_representations[0](indices=None).detach().numpy())

    with open(os.path.join(output_path, "relation_embeddings.npy"), "wb") as f:
        np.save(f, rel_representations[0](indices=None).detach().numpy())

    more_than_one_representation_message = (
        lambda ent_or_relation: f"There is more than one {ent_or_relation} representation for the trained model (see https://pykeen.readthedocs.io/en/stable/tutorial/first_steps.html?highlight=embedding#using-learned-embeddings). You may want to inspect the others by loading the PyTorch model using the model pickle."
    )

    if len(ent_representations) > 1:
        print(more_than_one_representation_message("entity"))
    if len(rel_representations) > 1:
        print(more_than_one_representation_message("relation"))

    # Save entity-to-ID and relation-to-ID mappings. IDs correspond to rows of the embeddings matrices.
    with open(os.path.join(output_path, "entities_to_ids.json"), "w") as f:
        json.dump(tf.entity_to_id, f)

    with open(os.path.join(output_path, "relations_to_ids.json"), "w") as f:
        json.dump(tf.relation_to_id, f)

    # Save plots
    if plot_losses:
        results.plot_losses()
        plt.savefig(os.path.join(output_path, "losses.png"), dpi=300)


if __name__ == "__main__":
    main()
