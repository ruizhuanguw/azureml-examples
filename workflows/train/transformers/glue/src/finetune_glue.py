import argparse
import logging
import numpy as np
import time
from typing import Any, List, Union, Dict, Callable

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

from azureml_adapter import set_environment_variables_for_nccl_backend, get_local_rank


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

    # set distributed learning env var and local_rank.
    # the first time training_args.device is called, it will init the process group
    set_environment_variables_for_nccl_backend()
    local_rank = get_local_rank()
    training_args.local_rank = local_rank

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bit training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

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
    run.log("time/epoch", (time.time() - start) / 60 / training_args.num_train_epochs)

    print("Evaluation...")

    trainer.evaluate()
