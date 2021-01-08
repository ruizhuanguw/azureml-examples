# description: Submit GLUE finetuning with Huggingface transformers library on Azure ML
"""Submit GLUE finetuning with Huggingface transformers library on Azure ML.

This script prepares the `src/finetune_glue.py` script to run in Azure ML.

To run this script you need:

    - An Azure ML Workspace
    - A ComputeTarget to train on (we recommend a GPU-based compute cluster)
    - Azure ML Environment:
        - create the required python environment by running the `aml_utils.py` script
        - This registers two environments "transformers-datasets-cpu" and "transformers-datasets-gpu"

Things to try:

    Different GLUE Tasks: "cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"
    Default: "cola"

    model_checkpoint: Huggingface provides pretrained models you can use, e.g.
    - "bert-base-cased"
    - "gpt2"
    - "xlnet-base-cased"
    - "roberta-base"
    See Huggingface documenation for full set of examples: https://huggingface.co/transformers/pretrained_models.html
    Default: "distilbert-base-uncased"

Note:
    
    Arguments passed to `src/finetune_glue.py` will override TrainingArguments:
    
    https://huggingface.co/transformers/main_classes/trainer.html#trainingarguments

"""
import argparse
from pathlib import Path
from azureml.core import Workspace  # connect to workspace
from azureml.core import ComputeTarget  # specify AzureML compute resources
from azureml.core import Experiment  # connect/create experiments
from azureml.core import Environment  # manage e.g. Python environments
from azureml.core import ScriptRunConfig  # prepare code, an run configuration
from azureml.core import Run  # used for type hints


def ort_transformers_environment():
    """Prepares Azure ML Environment with ORT + transformers library.

    This dockerfile is prepared using the ORT transformers example repo:

    https://github.com/microsoft/onnxruntime-training-examples/tree/master/huggingface-gpt2


    Return:
        Azure ML Environment with ort and huggingface libraries needed to perform
        GLUE finetuning task.
    """
    env = Environment('transformers-ort')
    env.docker.base_image = None
    env.docker.base_dockerfile = "./dockerfile"
    return env


def submit_glue_finetuning_to_aml(
    glue_task: str,
    model_checkpoint: str,
    environment: Environment,
    target: ComputeTarget,
    experiment: Experiment,
) -> Run:
    """Submit GLUE finetuning task to Azure ML.

    This method prepares the configuration (compute target and environment) together
    with the training code (see src) into a ScriptRunConfig, and submits it to Azure
    ML.

    Args:
        glue_task (str): Name of the GLUE finetuning task. One of: "cola", "mnli",
            "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli".
        model_checkpoint (str): Name of the transformers pretrained model to use
            for finetuning. See https://huggingface.co/transformers/pretrained_models.html
        environment (Environment): The Azure ML environment to use.
        target (ComputeTarget):  The Azure ML compute target to train on.
        experiment (Experiment):  The Azure ML experiment used to submit the run.

    Return:
        The Azure ML Run instance associated to this finetuning submission.
    """
    # set up script run configuration
    config = ScriptRunConfig(
        source_directory=str(Path(__file__).parent.joinpath("src")),
        script="finetune_glue.py",
        arguments=[
            "--output_dir",
            "outputs",
            "--task",
            glue_task,
            "--model_checkpoint",
            model_checkpoint,
            # training args
            "--num_train_epochs",
            5,
            "--learning_rate",
            2e-5,
            "--per_device_train_batch_size",
            16,
            "--per_device_eval_batch_size",
            16,
            "--disable_tqdm",
            True,
        ],
        compute_target=target,
        environment=environment,
    )

    # submit script to AML
    run = experiment.submit(config)
    run.set_tags(
        {
            "task": glue_task,
            "target": target.name,
            "environment": environment.name,
            "model": model_checkpoint,
        }
    )

    return run


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--glue_task", default="cola", help="Name of GLUE task used for finetuning."
    )
    parser.add_argument(
        "--model_checkpoint",
        default="distilbert-base-uncased",
        help="Pretrained transformers model name.",
    )
    args = parser.parse_args()

    print(
        f"Finetuning {args.glue_task} with model {args.model_checkpoint} on Azure ML..."
    )

    ws: Workspace = Workspace.from_config()

    target: ComputeTarget = ws.compute_targets["gpu-K80-2"]

    env: Environment = ort_transformers_environment()

    exp: Experiment = Experiment(ws, "transformers-glue-finetuning-ort")

    run: Run = submit_glue_finetuning_to_aml(
        glue_task=args.glue_task,
        model_checkpoint=args.model_checkpoint,  # try: "bert-base-uncased"
        environment=env,
        target=target,
        experiment=exp,
    )

    run.wait_for_completion(show_output=True)
