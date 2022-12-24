import torch
import numpy as np
from utils import flat_auroc_score, sharpen, mixup_augment, augment_smiles, to_torch
from mixmatch_loss import MixMatchLoss
from itertools import cycle


def mixmatch(
    x, y, u, model, augment_fn, embed_model, T=0.5, K=2, alpha=0.75, num_labels=2
):
    xb = np.array(embed_model.encode([augment_fn(smiles, k=1)[0] for smiles in x]))
    targets = np.array(y, dtype=int).reshape(-1)
    y = np.eye(num_labels)[targets]

    Ux, Uy = np.array([]), np.array([])
    for unlabelled_smiles in u:
        u_embed = embed_model.encode(augment_fn(unlabelled_smiles, k=K))
        u_pred = (
            sharpen(sum(map(lambda i: model(i), u_embed)) / u_embed.shape[0], T)
            .detach()
            .cpu()
            .numpy()
            .tolist()
        )
        qb = np.array([u_pred] * u_embed.shape[0])
        if Ux.shape[0] == 0:
            Ux = u_embed
            Uy = qb
        else:
            Ux = np.concatenate([Ux, u_embed], axis=0)
            Uy = np.concatenate([Uy, qb], axis=0)

    indices = np.random.shuffle(np.arange(len(xb) + len(Ux)))
    Wx = np.concatenate([Ux, xb], axis=0)[indices][0]
    Wy = np.concatenate([Uy, y], axis=0)[indices][0]

    X, p = mixup_augment(xb, Wx[: len(xb)], y, Wy[: len(xb)], alpha)
    U, q = mixup_augment(Ux, Wx[len(xb) :], Uy, Wy[len(xb) :], alpha)
    return X, U, p, q


def train_bert(
    labelled_dataloader,
    unlabelled_dataloader,
    val_dataloader,
    model_mlp,
    embed_model,
    args,
    set_device,
    optimizer,
    criterion,
):
    

    loss_fn = MixMatchLoss()
    best_model = None
    best_accuracy = 0.0
    train_loss_history, recall_train_history = [], []
    validation_loss_history, recall_validation_history = list(), list()
    for epoch in range(0, args.epoch):
        model_mlp.train()
        train_loss_scores = []
        training_acc_scores = []
        y_pred, y_true = list(), list()
        predictions = []
        for labelled_batch, unlabelled_batch in zip(
            cycle(labelled_dataloader), unlabelled_dataloader
        ):
            ## perform forward pass
            x, y = labelled_batch
            u, _ = unlabelled_batch
            x, u, p, q = mixmatch(
                x=x,
                y=y,
                u=u,
                model=model_mlp,
                augment_fn=augment_smiles,
                embed_model=embed_model,
                K=args.n_augment,
                num_labels=args.num_labels,
            )
            x, u, p, q = to_torch(x, u, p, q, device=set_device)

            ## compute loss and perform backward pass
            loss = loss_fn(x, u, p, q, model_mlp)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ## accumulate train loss
            train_loss_scores.append(loss.item())

        ## accumulate loss, recall, f1, precision per epoch
        train_loss_history.append((sum(train_loss_scores) / len(train_loss_scores)))
        # recall = flat_auroc_score(predictions, y_true)
        # recall_train_history.append(recall)
        print(f"Training =>  Epoch : {epoch+1} | Loss : {train_loss_history[-1]}")
        # | AUROC score: {recall_train_history[-1]}')

        model_mlp.eval()
        predictions = None
        with torch.no_grad():
            validation_loss_scores = list()
            y_true_val, y_pred_val = list(), list()

            ## perform validation pass
            for batch, targets in val_dataloader:
                ## perform forward pass
                batch = embed_model.encode(batch).to(set_device, dtype=torch.float32)
                pred = model_mlp(batch)
                predictions = pred
                preds = torch.max(pred, 1)[1]

                ## accumulate predictions per batch for the epoch
                y_pred_val += list([x.item() for x in preds.detach().cpu().numpy()])
                targets = torch.LongTensor([x.item() for x in list(targets)])
                y_true_val += list([x.item() for x in targets.detach().cpu().numpy()])

                ## computing validate loss
                loss = criterion(
                    pred.to(set_device), targets.to(set_device)
                )  ## compute loss

                ## accumulate validate loss
                validation_loss_scores.append(loss.item())

            ## accumulate loss, recall, f1, precision per epoch
            validation_loss_history.append(
                (sum(validation_loss_scores) / len(validation_loss_scores))
            )
            recall = flat_auroc_score(predictions, y_true_val)
            recall_validation_history.append(recall)

            print(
                f"Validation =>  Epoch : {epoch+1} | Loss : {validation_loss_history[-1]} | AUROC score: {recall_validation_history[-1]} "
            )

            if recall_validation_history[-1] > best_accuracy:
                best_accuracy = recall_validation_history[-1]
                print("Selecting the model...")
                best_model = model_mlp

    return best_model
