import pandas as pd
import torch
from keras_preprocessing.sequence import pad_sequences
from transformers import BertTokenizer
from tqdm import tqdm
import logging


from deepchem.molnet import (
    load_bbbp,
    load_bace_classification,
    load_tox21,
    load_clintox,
)
import numpy as np
from enumeration import SmilesEnumerator


## setting the threshold of logger to INFO
logging.basicConfig(filename="data_loader.log", level=logging.INFO)

## creating an object
logger = logging.getLogger()


MOLECULE_NET_DATASETS = {
    "bbbp": load_bbbp,
    "bace": load_bace_classification,
    "tox21": load_tox21,
    "clintox": load_clintox,
}


class MoleculeData:
    def __init__(self, dataset_name, max_sequence_length=512, debug=0):
        """
        Load dataset and bert tokenizer
        """
        self.debug = debug
        ## load data into memory
        tasks, datasets, transformers = MOLECULE_NET_DATASETS[dataset_name]()
        self.train_dataset, self.valid_dataset, self.test_dataset = datasets

        ## set max sequence length for model
        self.max_sequence_length = max_sequence_length
        ## get bert tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "shahrukhx01/smole-bert", do_lower_case=True
        )
        self.enumerator = SmilesEnumerator()
        self.dataset_name = dataset_name

    def train_val_test_split(self):
        """
        Separate out labels and texts
        """
        num_samples = 1_000_000
        if self.debug:
            print("Debug mode is enabled")
            num_samples = 100
        molecules_view1 = self.train_dataset.ids[:num_samples]
        print("Enumerating train SMILES")
        molecules_view2 = [
            self.enumerator.enumerate_smiles(input_smiles=smiles)
            for smiles in molecules_view1
        ]

        train_labels = None
        if self.dataset_name == "clintox":
            train_labels = self.train_dataset.y[:num_samples, 1]
        elif self.dataset_name == "tox21":
            train_labels = self.train_dataset.y[:num_samples, 11]
        else:
            train_labels = np.array(
                [int(label[0]) for label in self.train_dataset.y][:num_samples]
            )

        val_molecules_view1 = self.valid_dataset.ids
        val_molecules_view2 = [
            self.enumerator.enumerate_smiles(input_smiles=smiles)
            for smiles in val_molecules_view1
        ]
        val_labels = None
        if self.dataset_name == "clintox":
            val_labels = list(self.valid_dataset.y[:num_samples, 1])
        elif self.dataset_name == "tox21":
            val_labels = list(self.valid_dataset.y[:num_samples, 11])
        else:
            val_labels = [int(label[0]) for label in self.valid_dataset.y]

        test_molecules_view1 = self.test_dataset.ids
        test_molecules_view2 = [
            self.enumerator.enumerate_smiles(input_smiles=smiles)
            for smiles in test_molecules_view1
        ]
        test_labels = None
        if self.dataset_name == "clintox":
            test_labels = list(self.test_dataset.y[:num_samples, 1])
        elif self.dataset_name == "tox21":
            test_labels = list(self.test_dataset.y[:num_samples, 11])
        else:
            test_labels = [int(label[0]) for label in self.test_dataset.y]

        return (
            molecules_view1,
            molecules_view2,
            val_molecules_view1,
            val_molecules_view2,
            test_molecules_view1,
            test_molecules_view2,
            train_labels,
            val_labels,
            test_labels,
        )

    def preprocess(self, texts):
        """
        Add bert token (CLS and SEP) tokens to each sequence pre-tokenization
        """
        ## separate labels and texts before preprocessing
        # Adding CLS and SEP tokens at the beginning and end of each sequence for BERT
        texts_processed = ["[CLS] " + str(sequence) + " [SEP]" for sequence in texts]
        return texts_processed

    def tokenize(self, texts):
        """
        Use bert tokenizer to tokenize each sequence and post-process
        by padding or truncating to a fixed length
        """
        ## tokenize sequence
        tokenized_molecules = [self.tokenizer.tokenize(text) for text in tqdm(texts)]

        ## convert tokens to ids
        print("convert tokens to ids")
        text_ids = [
            self.tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenized_molecules)
        ]

        ## pad our text tokens for each sequence
        print("pad our text tokens for each sequence")
        text_ids_post_processed = pad_sequences(
            text_ids,
            maxlen=self.max_sequence_length,
            dtype="long",
            truncating="post",
            padding="post",
        )
        return text_ids_post_processed

    def create_attention_mask(self, text_ids):
        """
        Add attention mask for padding tokens
        """
        attention_masks = []
        # create a mask of 1s for each token followed by 0s for padding
        for seq in tqdm(text_ids):
            seq_mask = [float(i > 0) for i in seq]
            attention_masks.append(seq_mask)
        return attention_masks

    def process_molecules(self):
        """
        Apply preprocessing and tokenization pipeline of texts
        """
        ## perform the split
        (
            molecules_view1,
            molecules_view2,
            val_molecules_view1,
            val_molecules_view2,
            test_molecules_view1,
            test_molecules_view2,
            train_labels,
            val_labels,
            test_labels,
        ) = self.train_val_test_split()

        print("preprocessing texts")
        ## preprocess train, val, test texts
        train_molecules_view1 = self.preprocess(molecules_view1)
        train_molecules_view2 = self.preprocess(molecules_view2)
        val_molecules_processed_view1 = self.preprocess(val_molecules_view1)
        val_molecules_processed_view2 = self.preprocess(val_molecules_view2)
        test_molecules_processed_view1 = self.preprocess(test_molecules_view1)
        test_molecules_processed_view2 = self.preprocess(test_molecules_view2)

        del molecules_view1
        del molecules_view2
        del val_molecules_view1
        del val_molecules_view2
        del test_molecules_view1
        del test_molecules_view2

        ## preprocess train, val, test texts
        print("tokenizing train texts")
        train_ids_view1 = self.tokenize(train_molecules_view1)
        train_ids_view2 = self.tokenize(train_molecules_view2)
        print("tokenizing val texts")
        val_ids_view1 = self.tokenize(val_molecules_processed_view1)
        val_ids_view2 = self.tokenize(val_molecules_processed_view2)
        print("tokenizing test texts")
        test_ids_view1 = self.tokenize(test_molecules_processed_view1)
        test_ids_view2 = self.tokenize(test_molecules_processed_view2)

        del train_molecules_view1
        del train_molecules_view2
        del val_molecules_processed_view1
        del val_molecules_processed_view2
        del test_molecules_processed_view1
        del test_molecules_processed_view2

        ## create masks for train, val, test texts
        print("creating train attention masks for texts")
        train_masks_view1 = self.create_attention_mask(train_ids_view1)
        train_masks_view2 = self.create_attention_mask(train_ids_view2)
        print("creating val attention masks for texts")
        val_masks_view1 = self.create_attention_mask(val_ids_view1)
        val_masks_view2 = self.create_attention_mask(val_ids_view2)
        print("creating test attention masks for texts")
        test_masks_view1 = self.create_attention_mask(test_ids_view1)
        test_masks_view2 = self.create_attention_mask(test_ids_view2)
        return (
            train_ids_view1,
            train_ids_view2,
            val_ids_view1,
            val_ids_view2,
            test_ids_view1,
            test_ids_view2,
            train_masks_view1,
            train_masks_view2,
            val_masks_view1,
            val_masks_view2,
            test_masks_view1,
            test_masks_view2,
            train_labels,
            val_labels,
            test_labels,
        )

    def text_to_tensors(self):
        """
        Converting all the data into torch tensors
        """
        (
            train_ids_view1,
            train_ids_view2,
            val_ids_view1,
            val_ids_view2,
            test_ids_view1,
            test_ids_view2,
            train_masks_view1,
            train_masks_view2,
            val_masks_view1,
            val_masks_view2,
            test_masks_view1,
            test_masks_view2,
            train_labels,
            val_labels,
            test_labels,
        ) = self.process_molecules()

        print("converting all variables to tensors")
        ## convert inputs, masks and labels to torch tensors
        self.train_inputs_view1 = torch.tensor(train_ids_view1)
        self.train_inputs_view2 = torch.tensor(train_ids_view2)

        self.train_labels = torch.tensor(
            train_labels,
            dtype=torch.long,
        )
        self.train_masks_view1 = torch.tensor(train_masks_view1)
        self.train_masks_view2 = torch.tensor(train_masks_view2)

        self.validation_inputs_view1 = torch.tensor(val_ids_view1)
        self.validation_inputs_view2 = torch.tensor(val_ids_view2)
        self.validation_labels = torch.tensor(
            val_labels,
            dtype=torch.long,
        )
        self.validation_masks_view1 = torch.tensor(val_masks_view1)
        self.validation_masks_view2 = torch.tensor(val_masks_view2)

        self.test_inputs_view1 = torch.tensor(test_ids_view1)
        self.test_inputs_view2 = torch.tensor(test_ids_view2)
        self.test_labels = torch.tensor(
            test_labels,
            dtype=torch.long,
        )
        self.test_masks_view1 = torch.tensor(test_masks_view1)
        self.test_masks_view2 = torch.tensor(test_masks_view2)


if __name__ == "__main__":
    dataset_name = "bbbp"
    MoleculeData(dataset_name=dataset_name)
