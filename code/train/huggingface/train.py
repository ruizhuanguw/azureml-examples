import torch
from torch import optim
from transformers import AutoTokenizer, AutoModelForMaskedLM
import os

is_master = os.environ.get('AZUREML_PROCESS_NAME', 'main') in {'main', 'rank_0'}

def main():
    # initialize bert model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

    print(model)
    print("Finished Training")


if __name__ == "__main__":
    main()
