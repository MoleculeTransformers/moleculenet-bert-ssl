import torch
from sklearn.metrics import roc_auc_score
from torch import sigmoid
from torch.nn.functional import softmax
import numpy as np
from enumeration import SmilesEnumerator


enumerator = SmilesEnumerator()


def flat_auroc_score(preds, labels):
    """
    Function to calculate the roc_auc_score of our predictions vs labels
    """
    pred_flat = softmax(preds, dim=1)[:, 1]
    # labels_flat = np.argmax(labels, axis=1)
    return roc_auc_score(labels, pred_flat.detach().cpu().numpy())


def mixup_augment(embedding1, embedding2, label1, label2, alpha=1):
    lam = np.random.beta(alpha, alpha)
    embedding_output = lam * embedding1 + (1.0 - lam) * embedding2
    label_output = lam * label1 + (1.0 - lam) * label2
    return (embedding_output, label_output)


def get_perm(x, args):
    """get random permutation"""
    batch_size = x.size()[0]
    if args.cuda and torch.cuda.is_available():
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)
    return index


def sharpen(x, T):
    temp = x ** (1 / T)
    if temp.shape[0] == 2:
        temp = temp / temp.sum()
    else:
        temp = temp / temp.sum(axis=1, keepdims=True)
    return temp


def to_torch(*args, device="cuda"):
    convert_fn = lambda x: torch.from_numpy(x).to(device, dtype=torch.float32)
    return list(map(convert_fn, args))


def augment_smiles(smiles, k=5):
    return enumerator.smiles_enumeration(smiles, n_augment=k)
