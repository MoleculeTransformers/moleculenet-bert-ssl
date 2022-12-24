import torch
import numpy as np


class MixMatchLoss(torch.nn.Module):
    def __init__(self, lambda_u=75):
        super(MixMatchLoss, self).__init__()
        self.lambda_u = lambda_u
        self.xent = torch.nn.CrossEntropyLoss()
        self.mse = torch.nn.MSELoss()

    def forward(self, X, U, p, q, model):
        X_ = torch.cat([X, U], axis=0)
        preds = model(X_)
        loss = (self.xent(preds[: len(p)], torch.argmax(p, dim=1))) + (
            self.lambda_u * self.mse(preds[len(p) :], q)
        )
        return loss
