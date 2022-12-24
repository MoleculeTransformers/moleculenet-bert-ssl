from deepchem.molnet import load_bace_classification, load_bbbp
import numpy as np


import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from dataset import MolDataset, mol_collator

from args_parser import parse_args
import sys
import pandas as pd

_datasets = {"bace": load_bace_classification, "bbbp": load_bbbp}


def get_dataloaders(args):

    _, datasets, _ = _datasets.get(args.dataset_name)(reload=False)
    (train_dataset, valid_dataset, test_dataset) = datasets

    train_indices = []
    train_labels = [y[0] for y in train_dataset.y]
    label_df = pd.DataFrame(train_labels, columns=["labels"])
    if args.samples_per_class > 0:
        np.random.seed()
        tp = np.random.choice(
            list(label_df[label_df["labels"] == 1].index),
            args.samples_per_class,
            replace=False,
        )
        tn = np.random.choice(
            list(label_df[label_df["labels"] == 0].index),
            args.samples_per_class,
            replace=False,
        )
        train_indices = list(tp) + list(tn)

    np.random.seed()

    train_smiles = list(train_dataset.ids[train_indices])
    train_labels = [y[0] for y in train_dataset.y[train_indices]]

    val_smiles = list(valid_dataset.ids)
    val_labels = [y[0] for y in valid_dataset.y]

    test_smiles = list(test_dataset.ids)
    test_labels = [y[0] for y in test_dataset.y]

    labelled_data = MolDataset(train_smiles, train_labels)
    labelled_sampler = RandomSampler(labelled_data)
    labelled_dataloader = DataLoader(
        labelled_data,
        sampler=labelled_sampler,
        batch_size=args.batch_size,
        collate_fn=mol_collator,
    )

    unlabelled_indices = list(label_df.drop(train_indices, axis=0).index)
    unlabelled_data = MolDataset(
        list(train_dataset.ids[unlabelled_indices]),
        list([y[0] for y in train_dataset.y[unlabelled_indices]]),
    )
    unlabelled_sampler = RandomSampler(unlabelled_data)
    unlabelled_dataloader = DataLoader(
        unlabelled_data,
        sampler=unlabelled_sampler,
        batch_size=args.batch_size,
        collate_fn=mol_collator,
    )

    val_data = MolDataset(val_smiles, val_labels)
    val_sampler = RandomSampler(val_data)
    val_dataloader = DataLoader(
        val_data, sampler=val_sampler, batch_size=len(val_data), collate_fn=mol_collator
    )

    test_data = MolDataset(test_smiles, test_labels)
    test_sampler = RandomSampler(test_data)
    test_dataloader = DataLoader(
        test_data,
        sampler=test_sampler,
        batch_size=len(test_data),
        collate_fn=mol_collator,
    )

    return labelled_dataloader, unlabelled_dataloader, val_dataloader, test_dataloader
