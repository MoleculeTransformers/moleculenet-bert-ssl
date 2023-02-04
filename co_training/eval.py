import torch
from utils import flat_auroc_score


def eval_model(
    model_view1,
    model_view2,
    test_dataloader_view1,
    test_dataloader_view2,
    device,
    samples_per_class=-1,
    dataset_name="",
    out_file="eval_result_co_training.csv",
    posterior_threshold=0.8,
):
    ## tracking variables
    eval_accuracy_view1, eval_accuracy_view2, eval_accuracy_combined = 0, 0, 0
    nb_eval_steps = 0
    model_view1.eval()
    model_view2.eval()
    labels_view1, predictions_view1 = [], []
    labels_view2, predictions_view2 = [], []
    # Evaluate data for one epoch
    for batch_view1, batch_view2 in zip(
        test_dataloader_view1,
        test_dataloader_view2,
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
        predictions_view1 += list(logits_view1[0].detach().cpu().numpy())
        labels_view1 += list(b_labels_view1.to("cpu").numpy())

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
        predictions_view2 += list(logits_view2[0].detach().cpu().numpy())
        labels_view2 += list(b_labels_view2.to("cpu").numpy())

    tmp_eval_accuracy_view1 = flat_auroc_score(predictions_view1, labels_view1)
    tmp_eval_accuracy_view2 = flat_auroc_score(predictions_view2, labels_view2)
    tmp_eval_accuracy_combined = flat_auroc_score(
        (predictions_view1 + predictions_view2) / 2, labels_view2
    )
    print("Test auroc score view1: {}".format(eval_accuracy_view1 / nb_eval_steps))
    print("Test auroc score view2: {}".format(eval_accuracy_view2 / nb_eval_steps))
    print("Test auroc combined: {}".format(eval_accuracy_combined / nb_eval_steps))
    with open(out_file, "a+") as f:
        f.write(
            f"{dataset_name}, {samples_per_class}, {posterior_threshold}, {tmp_eval_accuracy_view1}, {tmp_eval_accuracy_view2}, {tmp_eval_accuracy_combined}\n"
        )
