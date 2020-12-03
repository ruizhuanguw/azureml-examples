import argparse
import os

from azureml.core import Run
from azureml.core import Workspace
try:
    from tokenizers import Tokenizer
except:
    from pip._internal import main as pip
    pip(['install',  'tokenizers'])

from tokenizers import Tokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.models import BPE
from tokenizers.normalizers import Lowercase, NFKC, Sequence
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer


def download_input(url):
    # Let's download the file and save it somewhere
    from requests import get
    with open('big.txt', 'wb') as big_f:
        response = get(url, )
        
        if response.status_code == 200:
            big_f.write(response.content)
        else:
            print("Unable to get the file: {}".format(response.reason))


def get_data_url():
    BIG_FILE_URL = 'https://raw.githubusercontent.com/dscape/spell/master/test/resources/big.txt'
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_url', required=False)
    args = parser.parse_args()
    return args.text_url if args.text_url else  BIG_FILE_URL
        

def create_tokenizer():
    #create the model for training
    # First we create an empty Byte-Pair Encoding model (i.e. not trained model)
    tokenizer = Tokenizer(BPE())

    # Then we enable lower-casing and unicode-normalization
    # The Sequence normalizer allows us to combine multiple Normalizer that will be
    # executed in order.
    tokenizer.normalizer = Sequence([
        NFKC(),
        Lowercase()
    ])

    # Our tokenizer also needs a pre-tokenizer responsible for converting the input to a ByteLevel representation.
    tokenizer.pre_tokenizer = ByteLevel()

    # And finally, let's plug a decoder so we can recover from a tokenized input to the original one
    tokenizer.decoder = ByteLevelDecoder()
    return tokenizer


def save_model(tokenizer):
    #save the model
    run = Run.get_context()
    model_dir = './model'
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    tokenizer.model.save(model_dir)
    run.upload_folder(name='model', path=model_dir)


def main():
    #get the training data location
    data_url = get_data_url()
    #download training data
    download_input(data_url)
    #create tokenizer for decoding.
    tokenizer = create_tokenizer()
    # We initialize our trainer, giving him the details about the vocabulary we want to generate
    trainer = BpeTrainer(vocab_size=25000, show_progress=True, initial_alphabet=ByteLevel.alphabet())

    #train the model
    tokenizer.train(trainer, ["big.txt"])

    print("Trained vocab size: {}".format(tokenizer.get_vocab_size()))
    save_model(tokenizer)


if __name__ == "__main__":
    main()
