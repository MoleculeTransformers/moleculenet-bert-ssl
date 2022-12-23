import torch
from utils import flat_auroc_score


def eval_model(
    model,
    test_dataloader,
    device,
    samples_per_class=-1,
    dataset_name="",
    out_file="eval_result.csv",
    n_augment=0,
):
    ## tracking variables
    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0

    ## evaluate data for one epoch
    for batch in test_dataloader:
        ## add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        ## unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        ## avoiding model's computation and storage of gradients -> saving memory and speeding up validation
        with torch.no_grad():
            # forward pass, calculate logit predictions
            logits = model(
                b_input_ids, token_type_ids=None, attention_mask=b_input_mask
            )

        ## move logits and labels to CPU
        logits = logits[0].detach().cpu().numpy()
        label_ids = b_labels.to("cpu").numpy()

        tmp_eval_accuracy = flat_auroc_score(logits, label_ids)

        eval_accuracy += tmp_eval_accuracy
        nb_eval_steps += 1
    print("Test AUROC: {}".format(eval_accuracy / nb_eval_steps))
    with open(out_file, "a+") as f:
        f.write(
            f"{dataset_name}, {samples_per_class}, {n_augment},{eval_accuracy / nb_eval_steps}\n"
        )
