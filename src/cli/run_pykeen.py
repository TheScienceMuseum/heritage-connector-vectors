import click
from pykeen.pipeline import pipeline_from_config
from pykeen.triples import TriplesFactory
import matplotlib.pyplot as plt
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

    if plot_losses:
        results.plot_losses()
        plt.savefig(os.path.join(output_path, "losses.png"), dpi=300)


if __name__ == "__main__":
    main()
