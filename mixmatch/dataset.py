import torch
from torch.utils.data import Dataset


class MolDataset(Dataset):
    def __init__(self, smiles, labels):

        self.smiles = smiles
        self.labels = labels

    def __getitem__(self, i):
        result = {}
        result["smiles"] = self.smiles[i]
        result["label"] = self.labels[i]
        return result

    def __len__(self):
        return len(self.labels)


def mol_collator(batch):
    assert all("smiles" in x for x in batch)
    assert all("label" in x for x in batch)
    return ([x["smiles"] for x in batch], torch.tensor([x["label"] for x in batch]))
