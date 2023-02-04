from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from utils import flat_auroc_score
from tqdm import tqdm, trange
from eval import eval_model
import torch
from itertools import cycle
from torch.nn import CrossEntropyLoss
from torch.nn.functional import softmax
from torch import nn
import math
import random

lambda_cot = 0.0


def adjust_lamda(epoch, config):
    epoch = epoch + 1
    global lambda_cot
    global lambda_diff
    if epoch <= config["warm_up"]:
        lambda_cot = config["lambda_cot_max"] * math.exp(
            -5 * (1 - epoch / config["warm_up"]) ** 2
        )
        lambda_diff = config["lambda_diff_max"] * math.exp(
            -5 * (1 - epoch / config["warm_up"]) ** 2
        )
    else:
        lambda_cot = config["lambda_cot_max"]
        lambda_diff = config["lambda_diff_max"]


def loss_cot(unlabelled_logits_view1, unlabelled_logits_view2, batch_size):
    # the Jensen-Shannon divergence between p1(x) and p2(x)
    S = nn.Softmax(dim=1)
    LS = nn.LogSoftmax(dim=1)
    a1 = 0.5 * (S(unlabelled_logits_view1) + S(unlabelled_logits_view2))
    loss1 = a1 * torch.log(a1)
    loss1 = -torch.sum(loss1)
    loss2 = S(unlabelled_logits_view1) * LS(unlabelled_logits_view1)
    loss2 = -torch.sum(loss2)
    loss3 = S(unlabelled_logits_view2) * LS(unlabelled_logits_view2)
    loss3 = -torch.sum(loss3)

    return (loss1 - 0.5 * (loss2 + loss3)) / batch_size


def train_co_training(
    model_view1,
    model_view2,
    optimizer_view1,
    scheduler_view1,
    optimizer_view2,
    scheduler_view2,
    train_dataloader_view1,
    train_dataloader_view2,
    unlabelled_data_view1,
    unlabelled_data_view2,
    validation_dataloader_view1,
    validation_dataloader_view2,
    test_dataloader_view1,
    test_dataloader_view2,
    epochs,
    device,
    eval_after=10,
    train_log=1,
    samples_per_class=-1,
    dataset_name="",
    num_labels=2,
    out_file="eval_result.csv",
    config={},
):
    t = []
    # Store our loss and accuracy for plotting
    unlabelled_predicted_train_ids_view1 = []
    unlabelled_predicted_train_ids_view2 = []

    unlabelled_predicted_train_masks_view1 = []
    unlabelled_predicted_train_masks_view2 = []

    unlabelled_predicted_train_labels_view1 = []
    unlabelled_predicted_train_labels_view2 = []

    train_loss_set = []
    unlabelled_loss_fct = CrossEntropyLoss()
    train_dataloader_combined_view1, train_dataloader_combined_view2 = None, None
    combined_dataset_view1, combined_dataset_view2 = None, None
    # trange is a tqdm wrapper around the normal python range
    for epoch in range(1, epochs + 1):

        if len(unlabelled_predicted_train_ids_view1) and len(
            unlabelled_predicted_train_ids_view2
        ):
            unlabelled_predicted_data_view1 = TensorDataset(
                torch.stack(unlabelled_predicted_train_ids_view1).detach().cpu(),
                torch.stack(unlabelled_predicted_train_masks_view1).detach().cpu(),
                torch.stack(unlabelled_predicted_train_labels_view1).detach().cpu(),
            )

            unlabelled_predicted_data_view2 = TensorDataset(
                torch.stack(unlabelled_predicted_train_ids_view2).detach().cpu(),
                torch.stack(unlabelled_predicted_train_masks_view2).detach().cpu(),
                torch.stack(unlabelled_predicted_train_labels_view2).detach().cpu(),
            )

            dataloaders = config["dataloader_wrapper_instance"]

            combined_dataset_view1 = torch.utils.data.ConcatDataset(
                [dataloaders.data_view_1, unlabelled_predicted_data_view1]
            )
            combined_dataset_view2 = torch.utils.data.ConcatDataset(
                [dataloaders.data_view_2, unlabelled_predicted_data_view2]
            )
            combined_sampler_view1 = RandomSampler(combined_dataset_view1)
            combined_sampler_view2 = RandomSampler(combined_dataset_view2)
            train_dataloader_combined_view1 = DataLoader(
                combined_dataset_view1,
                sampler=combined_sampler_view1,
                batch_size=config["batch_size"],
            )
            train_dataloader_combined_view2 = DataLoader(
                combined_dataset_view2,
                sampler=combined_sampler_view2,
                batch_size=config["batch_size"],
            )
        else:
            train_dataloader_combined_view1 = train_dataloader_view1
            train_dataloader_combined_view2 = train_dataloader_view2
            dataloaders = config["dataloader_wrapper_instance"]
            combined_dataset_view1, combined_dataset_view2 = (
                dataloaders.data_view_1,
                dataloaders.data_view_2,
            )
        print(f"Epoch: {epoch}")

        if epoch > 0 and epoch % eval_after == 0:
            print(f"Evaluating on test set")
            model_view1.eval()
            model_view2.eval()
            eval_model(
                model_view1=model_view1,
                model_view2=model_view2,
                test_dataloader_view1=test_dataloader_view1,
                test_dataloader_view2=test_dataloader_view2,
                device=device,
                samples_per_class=samples_per_class,
                dataset_name=dataset_name,
                out_file=out_file,
                posterior_threshold=config["posterior_threshold"],
            )

        ## set our model to training mode
        model_view1.train()
        model_view2.train()

        ## tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # train the model for one epoch
        for batch_view1, batch_view2 in zip(
            train_dataloader_combined_view1, train_dataloader_combined_view2
        ):
            indices_to_remove = []
            if train_log:
                print(f"combined dataset view1: {len(combined_dataset_view1)}")
                print(f"combined dataset view2: {len(combined_dataset_view2)}")
            ## move batch to GPU
            batch_view1 = tuple(t.to(device) for t in batch_view1)
            ## unpack the inputs from our dataloader
            b_input_ids_view1, b_input_mask_view1, b_labels_view1 = batch_view1

            ## forward pass
            outputs_view1 = model_view1(
                b_input_ids_view1,
                token_type_ids=None,
                attention_mask=b_input_mask_view1,
                labels=b_labels_view1,
            )
            loss_view1, logits_view1 = outputs_view1[:2]

            ## move batch to GPU
            batch_view2 = tuple(t.to(device) for t in batch_view2)
            ## unpack the inputs from our dataloader
            b_input_ids_view2, b_input_mask_view2, b_labels_view2 = batch_view2

            ## forward pass
            outputs_view2 = model_view2(
                b_input_ids_view2,
                token_type_ids=None,
                attention_mask=b_input_mask_view2,
                labels=b_labels_view2,
            )
            loss_view2, _ = outputs_view2[:2]
            unlabelled_batch_indices = random.sample(
                list(range(len(unlabelled_data_view1))),
                min(config["batch_size"], len(unlabelled_data_view1)),
            )
            if not len(unlabelled_batch_indices):
                ## stop training when no unlabelled data is left
                continue

            unlabelled_batch_view1 = (
                torch.stack(
                    [
                        unlabelled_data_view1[idx]["train_inputs"]
                        for idx in unlabelled_batch_indices
                    ]
                ),
                torch.stack(
                    [
                        unlabelled_data_view1[idx]["train_masks"]
                        for idx in unlabelled_batch_indices
                    ]
                ),
                torch.stack(
                    [
                        unlabelled_data_view1[idx]["train_labels"]
                        for idx in unlabelled_batch_indices
                    ]
                ),
            )

            unlabelled_batch_view2 = (
                torch.stack(
                    [
                        unlabelled_data_view2[idx]["train_inputs"]
                        for idx in unlabelled_batch_indices
                    ]
                ),
                torch.stack(
                    [
                        unlabelled_data_view2[idx]["train_masks"]
                        for idx in unlabelled_batch_indices
                    ]
                ),
                torch.stack(
                    [
                        unlabelled_data_view2[idx]["train_labels"]
                        for idx in unlabelled_batch_indices
                    ]
                ),
            )

            ## unpack the inputs from our dataloader
            (
                unlabelled_input_ids_view1,
                unlabelled_input_mask_view1,
                _,
            ) = unlabelled_batch_view1
            (
                unlabelled_input_ids_view2,
                unlabelled_input_mask_view2,
                _,
            ) = unlabelled_batch_view2
            # labelled_loss, labelled_logits = outputs[:2]
            unlabelled_outputs_view1 = model_view1(
                unlabelled_input_ids_view1.to(device),
                token_type_ids=None,
                attention_mask=unlabelled_input_mask_view1.to(device),
                labels=None,
            )
            unlabelled_logits_view1 = unlabelled_outputs_view1.logits

            unlabelled_outputs_view2 = model_view1(
                unlabelled_input_ids_view2.to(device),
                token_type_ids=None,
                attention_mask=unlabelled_input_mask_view2.to(device),
                labels=None,
            )
            unlabelled_logits_view2 = unlabelled_outputs_view2.logits

            unlabelled_posteriors_view1 = softmax(unlabelled_logits_view1, dim=1)
            unlabelled_posteriors_view2 = softmax(unlabelled_logits_view2, dim=1)
            for idx, (label, posterior) in enumerate(
                zip(
                    torch.max(unlabelled_posteriors_view1, dim=1).indices,
                    torch.max(unlabelled_posteriors_view1, dim=1).values,
                )
            ):
                if posterior >= config["posterior_threshold"]:
                    unlabelled_predicted_train_ids_view1.append(
                        unlabelled_data_view1[unlabelled_batch_indices[idx]][
                            "train_inputs"
                        ]
                    )
                    unlabelled_predicted_train_masks_view1.append(
                        unlabelled_data_view1[unlabelled_batch_indices[idx]][
                            "train_masks"
                        ]
                    )
                    unlabelled_predicted_train_labels_view1.append(label)

                    unlabelled_predicted_train_ids_view2.append(
                        unlabelled_data_view2[unlabelled_batch_indices[idx]][
                            "train_inputs"
                        ]
                    )
                    unlabelled_predicted_train_masks_view2.append(
                        unlabelled_data_view2[unlabelled_batch_indices[idx]][
                            "train_masks"
                        ]
                    )
                    unlabelled_predicted_train_labels_view2.append(label)
                    indices_to_remove.append(unlabelled_batch_indices[idx])

            for idx, (label, posterior) in enumerate(
                zip(
                    torch.max(unlabelled_posteriors_view2, dim=1).indices,
                    torch.max(unlabelled_posteriors_view2, dim=1).values,
                )
            ):
                if posterior >= config["posterior_threshold"]:
                    unlabelled_predicted_train_ids_view1.append(
                        unlabelled_data_view1[unlabelled_batch_indices[idx]][
                            "train_inputs"
                        ]
                    )
                    unlabelled_predicted_train_masks_view1.append(
                        unlabelled_data_view1[unlabelled_batch_indices[idx]][
                            "train_masks"
                        ]
                    )
                    unlabelled_predicted_train_labels_view1.append(label)

                    unlabelled_predicted_train_ids_view2.append(
                        unlabelled_data_view2[unlabelled_batch_indices[idx]][
                            "train_inputs"
                        ]
                    )
                    unlabelled_predicted_train_masks_view2.append(
                        unlabelled_data_view2[unlabelled_batch_indices[idx]][
                            "train_masks"
                        ]
                    )
                    unlabelled_predicted_train_labels_view2.append(label)
                    indices_to_remove.append(unlabelled_batch_indices[idx])
            for idx_remove in list(set(indices_to_remove)):
                try:
                    del unlabelled_data_view1[idx_remove]
                    del unlabelled_data_view2[idx_remove]
                except:
                    continue
            ## reset the gradients
            optimizer_view1.zero_grad()
            optimizer_view2.zero_grad()

            # semi-supervised loss
            supervised_loss = loss_view1 + loss_view2
            unlabelled_cotraining_loss = loss_cot(
                unlabelled_logits_view1, unlabelled_logits_view2, config["batch_size"]
            )

            loss = supervised_loss + (lambda_cot * unlabelled_cotraining_loss)

            ## backward pass
            loss.backward()

            ## adjust lambda weight loss term for co-training
            adjust_lamda(epoch=epoch, config=config)

            ## update parameters and take a step using the computed gradient
            optimizer_view1.step()
            optimizer_view2.step()

            ## update the learning rate.
            scheduler_view1.step()
            scheduler_view2.step()

            ## update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids_view1.size(0)
            nb_tr_steps += 1
        if train_log:
            print("Train loss: {}".format(tr_loss / max(nb_tr_steps, 1.0)))

        # Put model in evaluation mode to evaluate loss on the validation set
        model_view1.eval()
        model_view2.eval()

        # Tracking variables
        eval_accuracy_view1, eval_accuracy_view2, eval_accuracy_combined = 0, 0, 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch_view1, batch_view2 in zip(
            validation_dataloader_view1,
            validation_dataloader_view2,
        ):
            # add batch to GPU
            batch_view1 = tuple(t.to(device) for t in batch_view1)
            # unpack the inputs from our dataloader
            b_input_ids_view1, b_input_mask_view1, b_labels_view1 = batch_view1
            # avoiding model's computation and storage of gradients -> saving memory and speeding up validation
            with torch.no_grad():
                # forward pass, calculate logit predictions
                logits_view1 = model_view1(
                    b_input_ids_view1,
                    token_type_ids=None,
                    attention_mask=b_input_mask_view1,
                )

            # Move logits and labels to CPU
            logits_view1 = logits_view1[0].detach().cpu().numpy()
            label_ids_view1 = b_labels_view1.to("cpu").numpy()

            tmp_eval_accuracy_view1 = flat_auroc_score(logits_view1, label_ids_view1)

            eval_accuracy_view1 += tmp_eval_accuracy_view1
            # add batch to GPU
            batch_view2 = tuple(t.to(device) for t in batch_view2)
            # unpack the inputs from our dataloader
            b_input_ids_view2, b_input_mask_view2, b_labels_view2 = batch_view2
            # avoiding model's computation and storage of gradients -> saving memory and speeding up validation
            with torch.no_grad():
                # forward pass, calculate logit predictions
                logits_view2 = model_view2(
                    b_input_ids_view2,
                    token_type_ids=None,
                    attention_mask=b_input_mask_view2,
                )

            # Move logits and labels to CPU
            logits_view2 = logits_view2[0].detach().cpu().numpy()
            label_ids_view2 = b_labels_view2.to("cpu").numpy()

            tmp_eval_accuracy_view2 = flat_auroc_score(logits_view2, label_ids_view2)

            eval_accuracy_view2 += tmp_eval_accuracy_view2
            # combine logits from both view models
            assert (label_ids_view1 == label_ids_view2).all()
            tmp_eval_accuracy_combined = flat_auroc_score(
                (logits_view1 + logits_view2) / 2, label_ids_view2
            )

            eval_accuracy_combined += tmp_eval_accuracy_combined
            nb_eval_steps += 1
        # model.save_pretrained("./model")
        if train_log:
            print(
                "Validation auroc score view1: {}".format(
                    eval_accuracy_view1 / nb_eval_steps
                )
            )
            print(
                "Validation auroc score view2: {}".format(
                    eval_accuracy_view2 / nb_eval_steps
                )
            )
            print(
                "Validation auroc score combined: {}".format(
                    eval_accuracy_combined / nb_eval_steps
                )
            )

    return None  # model
