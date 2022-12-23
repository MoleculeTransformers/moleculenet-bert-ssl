import torch
from model import BERTClassifier
from config import BertOptimConfig
from train import train_model, train_pseudo_label_model
from eval import eval_model
from data_loader import MoleculeDataLoader
import argparse

parser = argparse.ArgumentParser(
    description="PyTorch Implementation of Semi Supervised Molecule Property Prediction with BERT"
)
parser.add_argument(
    "--model-name-or-path",
    type=str,
    default="shahrukhx01/smole-bert",
    metavar="M",
    help="name of the pre-trained transformer model from hf hub",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=20,
    metavar="B",
    help="size of the dataloader batch",
)

parser.add_argument(
    "--epochs",
    type=int,
    default=10,
    metavar="N",
    help="number of epochs to train (default: 10)",
)


parser.add_argument(
    "--dataset-name",
    type=str,
    default="bace",
    metavar="D",
    help="name of the molecule net dataset (default: bace) all: bace, bbbp",
)

parser.add_argument(
    "--num-labels",
    type=int,
    default=2,
    metavar="L",
    help="number of labels of the train dataset (default: 2)",
)

parser.add_argument(
    "--eval-after",
    type=int,
    default=10,
    metavar="EA",
    help="number of epochs after which model is evaluated on test set (default: 10)",
)

parser.add_argument(
    "--debug",
    type=int,
    default=0,
    metavar="DB",
    help="flag to enable debug mode for dev (default: 0)",
)

parser.add_argument(
    "--samples-per-class",
    type=int,
    default=-1,
    metavar="SPC",
    help="no. of samples per class label to sample for SSL (default: 250)",
)

parser.add_argument(
    "--train-log",
    type=int,
    default=1,
    metavar="TL",
    help="flag to enable logging training metrics (default: 1)",
)

parser.add_argument(
    "--train-ssl",
    type=int,
    default=0,
    metavar="TSSL",
    help="flag to enable SSL-based training (default: 0)",
)

parser.add_argument(
    "--out-file",
    type=str,
    default="eval_result.csv",
    metavar="OF",
    help="outpul file for logging metrics",
)
parser.add_argument(
    "--n-augment",
    type=int,
    default=0,
    metavar="NAUG",
    help="number of enumeration augmentations",
)


args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = MoleculeDataLoader(
        dataset_name=args.dataset_name,
        batch_size=args.batch_size,
        debug=args.debug,
        n_augment=args.n_augment,
        samples_per_class=args.samples_per_class,
        model_name_or_path=args.model_name_or_path,
    )
    if args.train_ssl:
        data_loaders.create_semi_supervised_loaders(
            samples_per_class=args.samples_per_class
        )
    else:
        data_loaders.create_supervised_loaders(samples_per_class=args.samples_per_class)

    model = BERTClassifier(num_labels=args.num_labels).get_model(
        model_name_or_path=args.model_name_or_path
    )
    optim_config = BertOptimConfig(
        model=model,
        train_dataloader=data_loaders.labelled_dataloader
        if args.train_ssl
        else data_loaders.train_dataloader,
        epochs=args.epochs,
    )

    ## execute the training routine
    if args.train_ssl:
        model = train_pseudo_label_model(
            model=model,
            optimizer=optim_config.optimizer,
            scheduler=optim_config.scheduler,
            labelled_dataloader=data_loaders.labelled_dataloader,
            unlabelled_dataloader=data_loaders.unlabelled_dataloader,
            validation_dataloader=data_loaders.validation_dataloader,
            test_dataloader=data_loaders.test_dataloader,
            epochs=args.epochs,
            device=device,
            eval_after=args.eval_after,
            train_log=args.train_log,
            samples_per_class=args.samples_per_class,
            dataset_name=args.dataset_name,
            num_labels=args.num_labels,
            out_file=args.out_file,
            n_augment=args.n_augment,
        )
    else:
        model = train_model(
            model=model,
            optimizer=optim_config.optimizer,
            scheduler=optim_config.scheduler,
            train_dataloader=data_loaders.train_dataloader,
            validation_dataloader=data_loaders.validation_dataloader,
            test_dataloader=data_loaders.test_dataloader,
            epochs=args.epochs,
            device=device,
            eval_after=args.eval_after,
            train_log=args.train_log,
            samples_per_class=args.samples_per_class,
            dataset_name=args.dataset_name,
            out_file=args.out_file,
            n_augment=args.n_augment,
        )

    ## test model performance on unseen test set
    # eval_model(model=model, test_dataloader=data_loaders.test_dataloader, device=device)
