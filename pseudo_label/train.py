from utils import flat_auroc_score
from tqdm import tqdm, trange
from eval import eval_model
import torch
from itertools import cycle
from torch.nn import CrossEntropyLoss


def train_model(
    model,
    optimizer,
    scheduler,
    train_dataloader,
    validation_dataloader,
    test_dataloader,
    epochs,
    device,
    eval_after=10,
    train_log=1,
    samples_per_class=-1,
    dataset_name="",
    out_file="eval_result.csv",
    n_augment=0,
):
    t = []

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # trange is a tqdm wrapper around the normal python range
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        if epoch > 0 and epoch % eval_after == 0:
            print(f"Evaluating on test set")
            eval_model(
                model=model,
                test_dataloader=test_dataloader,
                device=device,
                samples_per_class=samples_per_class,
                dataset_name=dataset_name,
                out_file=out_file,
                n_augment=n_augment,
            )

        ## set our model to training mode
        model.train()

        ## tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train the model for one epoch
        for step, batch in enumerate(train_dataloader):
            ## move batch to GPU
            batch = tuple(t.to(device) for t in batch)
            ## unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            ## reset the gradients
            optimizer.zero_grad()
            ## forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            loss, logits = outputs[:2]
            train_loss_set.append(loss.item())
            ## backward pass
            loss.backward()
            ## update parameters and take a step using the computed gradient
            optimizer.step()

            ## update the learning rate.
            scheduler.step()

            ## update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        if train_log:
            print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # avoiding model's computation and storage of gradients -> saving memory and speeding up validation
            with torch.no_grad():
                # forward pass, calculate logit predictions
                logits = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tmp_eval_accuracy = flat_auroc_score(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        model.save_pretrained("./model")
        if train_log:
            print("Validation auroc score: {}".format(eval_accuracy / nb_eval_steps))

    return model


def train_pseudo_label_model(
    model,
    optimizer,
    scheduler,
    labelled_dataloader,
    unlabelled_dataloader,
    validation_dataloader,
    test_dataloader,
    epochs,
    device,
    eval_after=10,
    train_log=1,
    samples_per_class=-1,
    dataset_name="",
    num_labels=2,
    out_file="eval_result.csv",
    n_augment=0,
):
    t = []
    T1 = 10
    T2 = 60
    alpha = 0
    af = 0.3

    # Store our loss and accuracy for plotting
    train_loss_set = []
    unlabelled_loss_fct = CrossEntropyLoss()
    # trange is a tqdm wrapper around the normal python range
    for epoch in range(1, epochs + 1):
        print(f"Epoch: {epoch}")
        if epoch > 0 and epoch % eval_after == 0:
            print(f"Evaluating on test set")
            eval_model(
                model=model,
                test_dataloader=test_dataloader,
                device=device,
                samples_per_class=samples_per_class,
                dataset_name=dataset_name,
                out_file=out_file,
                n_augment=n_augment,
            )

        ## set our model to training mode
        model.train()

        ## tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train the model for one epoch
        for labelled_batch, unlabelled_batch in zip(
            cycle(labelled_dataloader), unlabelled_dataloader
        ):
            ## move batch to GPU
            labelled_batch = tuple(t.to(device) for t in labelled_batch)
            ## unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = labelled_batch
            ## reset the gradients
            optimizer.zero_grad()
            ## forward pass
            outputs = model(
                b_input_ids,
                token_type_ids=None,
                attention_mask=b_input_mask,
                labels=b_labels,
            )
            labelled_loss, labelled_logits = outputs[:2]
            # train_loss_set.append(loss.item())

            ## process unlabelled batch
            ## move batch to GPU
            unlabelled_batch = tuple(t.to(device) for t in unlabelled_batch)
            ## unpack the inputs from our dataloader
            unlabelled_input_ids, unlabelled_input_mask, _ = unlabelled_batch
            labelled_loss, labelled_logits = outputs[:2]

            ## reset the gradients
            optimizer.zero_grad()

            # unlabeled forward pass
            if epoch > T1:
                alpha = (epoch - T1) / (T2 - T1) * af
                if epoch > T2:
                    alpha = af
            # end_if

            unlabelled_outputs = model(
                unlabelled_input_ids,
                token_type_ids=None,
                attention_mask=unlabelled_input_mask,
                labels=None,
            )

            unlabelled_logits = unlabelled_outputs.logits

            pseudo_labels = torch.argmax(unlabelled_logits, dim=1)

            # semi-supervised loss
            loss = labelled_loss + alpha * unlabelled_loss_fct(
                unlabelled_logits.view(-1, num_labels), pseudo_labels.view(-1)
            )

            ## backward pass
            loss.backward()
            ## update parameters and take a step using the computed gradient
            optimizer.step()

            ## update the learning rate.
            scheduler.step()

            ## update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1
        if train_log:
            print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # avoiding model's computation and storage of gradients -> saving memory and speeding up validation
            with torch.no_grad():
                # forward pass, calculate logit predictions
                logits = model(
                    b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                )

            # Move logits and labels to CPU
            logits = logits[0].detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()

            tmp_eval_accuracy = flat_auroc_score(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
        model.save_pretrained("./model")
        if train_log:
            print("Validation auroc score: {}".format(eval_accuracy / nb_eval_steps))

    return model
