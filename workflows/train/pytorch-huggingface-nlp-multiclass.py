# description: train a pytorch CNN model on mnist data

# imports

from pathlib import Path
from azureml.core import Workspace
from azureml.core import ScriptRunConfig, Experiment, Environment

# get workspace
ws = Workspace.from_config()

# get root of git repo
prefix = Path("../..") # Path(git.Repo(".", search_parent_directories=True).working_tree_dir)

# training script
script_dir = prefix.joinpath("code", "train", "huggingface")
script_name = "train.py"

# environment file
environment_file = prefix.joinpath("environments", "huggingface.yml")

# azure ml settings
environment_name = "pytorch-huggingface-example"
experiment_name = "pytorch-huggingface-example"
compute_target = "gpu-cluster"

# script arguments
arguments = ["--epochs", 2]

# create environment
env = Environment.from_conda_specification(environment_name, environment_file)

# create job config
src = ScriptRunConfig(
    source_directory=script_dir,
    script=script_name,
    arguments=arguments,
    environment=env,
    compute_target=compute_target,
)

# submit job
run = Experiment(ws, experiment_name).submit(src)
print(run)
# run.wait_for_completion(show_output=True)
