"""Automatic hyperparameter optimization with Azure ML HyperDrive library.

This submits a HyperDrive experiment to optimize for a set of hyperparameters
using the `code/train/transformers/glue/train.py` script to submit individual
finetuning runs.

We use:

- Early termination policy to halt "poorly performing" runs
- Concurrency, that allows us to parellelize individual finetuning runs
"""
import argparse
import numpy as np
from azureml.core import Workspace  # connect to workspace
from azureml.core import Experiment  # connect/create experiments
from azureml.core import Environment  # manage e.g. Python environments
from azureml.core import ScriptRunConfig  # prepare code, an run configuration

# hyperdrive imports
from azureml.train.hyperdrive import (
    RandomParameterSampling,
    BayesianParameterSampling,
    TruncationSelectionPolicy,
    MedianStoppingPolicy,
    HyperDriveConfig
)
from azureml.train import hyperdrive


def create_transformers_environment(workspace):
    """Register transformers gpu-base image to workspace."""
    env_name = "transformers-gpu"
    if env_name not in workspace.environments:

        # get root of git repo
        prefix = Path(__file__).parent.parent.parent.absolute()
        pip_requirements_path = prefix.joinpath("environments", "transformers-requirements.txt")
        print(f"Create Azure ML Environment {env_name} from {pip_requirements_path}")
        env = Environment.from_pip_requirements(
            name=env_name, file_path=pip_requirements_path,
        )
        env.docker.base_image = "mcr.microsoft.com/azureml/intelmpi2018.3-cuda10.0-cudnn7-ubuntu16.04"
        return env

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--compute_target_name', default="gpu-K80-2")
    parser.add_argument('--environment_name', default="transformers-datasets-gpu")
    parser.add_argument('--model_checkpoint', default="distilbert-base-uncased")
    parser.add_argument('--glue_task', default="cola")
    args = parser.parse_args()

    ws = Workspace.from_config()

    env = create_transformers_environment(ws)

    # get compute target
    target = ws.compute_targets[args.compute_target_name]

    # get environment
    env = ws.environments[args.environment_name]

    # set up script run configuration
    config = ScriptRunConfig(
        source_directory="src",
        script="run_glue.py",
        arguments=[
            "--output_dir", "outputs",
            "--task", args.glue_task,
            "--model_checkpoint", args.model_checkpoint,
            
            # training args
            "--evaluation_strategy", "steps",
            "--eval_steps", 200,
            "--learning_rate", 2e-5,
            "--per_device_train_batch_size", 16,
            "--per_device_eval_batch_size", 16,
            "--num_train_epochs", 5,
            "--weight_decay", 0.01,
            "--disable_tqdm", True,
        ],
        compute_target=target,
        environment=env,
    )

    # set up hyperdrive search space
    convert_base = lambda x: float(np.log(x))
    search_space = {
        "--learning_rate": hyperdrive.loguniform(convert_base(1e-6), convert_base(5e-2)),  # NB. loguniform on [exp(min), exp(max)]
        "--weight_decay": hyperdrive.uniform(5e-3, 15e-2),
        "--per_device_train_batch_size": hyperdrive.choice([16, 32]),
    }

    hyperparameter_sampling = RandomParameterSampling(search_space)

    policy = TruncationSelectionPolicy(
        truncation_percentage=50,
        evaluation_interval=2,
        delay_evaluation=0
    )

    hyperdrive_config = HyperDriveConfig(
        run_config=config,
        hyperparameter_sampling=hyperparameter_sampling,
        policy=policy,
        primary_metric_name="eval_matthews_correlation",
        primary_metric_goal=hyperdrive.PrimaryMetricGoal.MAXIMIZE,
        max_total_runs=20,
        max_concurrent_runs=8,
    )

    run = Experiment(ws, "transformers-glue-finetuning-hyperdrive").submit(hyperdrive_config)
    print(run.get_portal_url())