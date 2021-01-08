import argparse
import logging
import numpy as np
import time
from typing import Any, List, Union, Dict, Callable
import torch

# from dataclasses import dataclass, field
from datasets import Metric
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    HfArgumentParser,
    TrainingArguments,
)
from glue_datasets import (
    load_encoded_glue_dataset,
    num_labels_from_task,
    load_metric_from_task,
)

# Azure ML imports - could replace this with e.g. wandb or mlflow
from transformers.integrations import AzureMLCallback
from azureml.core import Run

logger = logging.getLogger(__name__)

from distributed_utils import set_environment_variables_for_nccl_backend, get_local_rank


def construct_compute_metrics_function(task: str) -> Callable[[EvalPrediction], Dict]:
    metric = load_metric_from_task(task)

    if task != "stsb":

        def compute_metrics_function(eval_pred: EvalPrediction) -> Dict:
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            return metric.compute(predictions=predictions, references=labels)

    else:

        def compute_metrics_function(eval_pred: EvalPrediction) -> Dict:
            predictions, labels = eval_pred
            predictions = predictions[:, 0]
            return metric.compute(predictions=predictions, references=labels)

    return compute_metrics_function


if __name__ == "__main__":

    parser = HfArgumentParser(TrainingArguments)
    parser.add_argument("--task", default="cola", help="name of GLUE task to compute")
    parser.add_argument("--model_checkpoint", default="distilbert-base-uncased")
    training_args, args = parser.parse_args_into_dataclasses()

    for k, v in os.environ.items():
        print(k, v)

    # # Setup CUDA, GPU & distributed training
    # local_rank = get_local_rank()
    # print(f"Local rank: {local_rank}")
    # if local_rank == -1:
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     args.n_gpu = torch.cuda.device_count()
    # else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
    #     set_environment_variables_for_nccl_backend()
    #     torch.cuda.set_device(local_rank)
    #     device = torch.device("cuda", local_rank)
    #     torch.distributed.init_process_group(backend="nccl")
    #     # args.n_gpu = 1
    # args.device = device

    task: str = args.task.lower()

    tokenizer = AutoTokenizer.from_pretrained(args.model_checkpoint, use_fast=True)

    encoded_dataset_train, encoded_dataset_eval = load_encoded_glue_dataset(
        task=task, tokenizer=tokenizer
    )

    num_labels = num_labels_from_task(task)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_checkpoint, num_labels=num_labels
    )

    compute_metrics = construct_compute_metrics_function(args.task)

    trainer = Trainer(
        model,
        training_args,
        callbacks=[AzureMLCallback()],
        train_dataset=encoded_dataset_train,
        eval_dataset=encoded_dataset_eval,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("Training...")

    run = Run.get_context()  # get handle on Azure ML run
    start = time.time()
    trainer.train()
    # run.log(f"time/epoch(rank{local_rank})", (time.time() - start) / 60 / training_args.num_train_epochs)
    run.log(f"time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs)

    print("Evaluation...")
    trainer.evaluate()
