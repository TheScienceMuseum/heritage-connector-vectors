import click
from pykeen.pipeline import pipeline_from_config
import matplotlib.pyplot as plt
import numpy as np
import os
import json
import pathlib
from utils import get_timestamp, triplesfactory_from_tsv


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
    default=True,
    help="Specify device to train on. Default is GPU; to train on CPU, pass `--cpu`.",
)
@click.option("--plot_losses/--no_plot_losses", is_flag=True, default=True)
@click.option("-s", "--random_seed", type=int, default=100)
@click.option(
    "--from_checkpoint",
    type=click.Path(exists=True, dir_okay=False),
    required=False,
    help="Path to load a checkpoint from, and start training at that checkpoint. See https://pykeen.readthedocs.io/en/stable/tutorial/checkpoints.html.",
)
@click.option("--save_checkpoint/--no_save_checkpoint", is_flag=True, default=True)
def main(
    input_data_path,
    output_path,
    config_path,
    gpu,
    plot_losses,
    random_seed,
    from_checkpoint,
    save_checkpoint,
):

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
    tf = triplesfactory_from_tsv(input_data_path, create_inverse_triples=False)

    if save_checkpoint:
        checkpoint_dir = "./data/checkpoints"
        checkpoint_name = f"{config['metadata']['title']}.pt"

    if from_checkpoint:
        path = pathlib.Path(from_checkpoint)
        checkpoint_dir, checkpoint_name = path.parent, path.name

    if from_checkpoint or save_checkpoint:
        if "training_kwargs" in config["pipeline"].keys():
            config["pipeline"]["training_kwargs"].update(
                {
                    "checkpoint_name": checkpoint_name,
                    "checkpoint_directory": checkpoint_dir,
                }
            )
        else:
            config["pipeline"]["training_kwargs"] = {
                "checkpoint_name": checkpoint_name,
                "checkpoint_directory": checkpoint_dir,
            }

    pipeline_kwargs = dict(
        training=tf,
        testing=tf,
        device="gpu" if gpu else "cpu",
        random_seed=random_seed,
    )

    results = pipeline_from_config(config, **pipeline_kwargs)
    results.save_to_directory(output_path)

    # Save embeddings matrices
    ent_representations = results.model.entity_representations
    rel_representations = results.model.relation_representations

    with open(os.path.join(output_path, "entity_embeddings.npy"), "wb") as f:
        np.save(f, ent_representations[0](indices=None).detach().cpu().numpy())

    with open(os.path.join(output_path, "relation_embeddings.npy"), "wb") as f:
        np.save(f, rel_representations[0](indices=None).detach().cpu()().numpy())

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
