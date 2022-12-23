import torch
from model import BERTClassifier
from config import BertOptimConfig
from train import train_co_training
from data_loader import MoleculeDataLoader
import argparse

parser = argparse.ArgumentParser(
    description="PyTorch Implementation of Semi Supervised Molecule Property Prediction with BERT"
)
parser.add_argument("--lambda_cot_max", default=10, type=int)
parser.add_argument("--lambda_diff_max", default=0.5, type=float)
parser.add_argument("--warm_up", default=15.0, type=float)
parser.add_argument("--posterior-threshold", default=0.9, type=float)
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

args = parser.parse_args()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_loaders = MoleculeDataLoader(
        dataset_name=args.dataset_name, batch_size=args.batch_size, debug=args.debug
    )
    if args.train_ssl:
        data_loaders.create_semi_supervised_loaders(
            samples_per_class=args.samples_per_class
        )
    else:
        data_loaders.create_supervised_loaders(samples_per_class=args.samples_per_class)

    model_view1 = BERTClassifier(num_labels=args.num_labels).get_model(
        model_name_or_path=args.model_name_or_path
    )
    model_view2 = BERTClassifier(num_labels=args.num_labels).get_model(
        model_name_or_path=args.model_name_or_path
    )
    optim_config_view1 = BertOptimConfig(
        model=model_view1,
        train_dataloader=data_loaders.view_1_dataloader,
        epochs=args.epochs,
    )

    optim_config_view2 = BertOptimConfig(
        model=model_view2,
        train_dataloader=data_loaders.view_2_dataloader,
        epochs=args.epochs,
    )

    ## execute the training routine
    if args.train_ssl:
        model = train_co_training(
            model_view1=model_view1,
            model_view2=model_view2,
            optimizer_view1=optim_config_view1.optimizer,
            scheduler_view1=optim_config_view1.scheduler,
            optimizer_view2=optim_config_view2.optimizer,
            scheduler_view2=optim_config_view2.scheduler,
            train_dataloader_view1=data_loaders.view_1_dataloader,
            train_dataloader_view2=data_loaders.view_2_dataloader,
            unlabelled_data_view1=data_loaders.unlabelled_data_view1,
            unlabelled_data_view2=data_loaders.unlabelled_data_view2,
            validation_dataloader_view1=data_loaders.validation_dataloader_view1,
            validation_dataloader_view2=data_loaders.validation_dataloader_view2,
            test_dataloader_view1=data_loaders.test_dataloader_view1,
            test_dataloader_view2=data_loaders.test_dataloader_view2,
            epochs=args.epochs,
            device=device,
            eval_after=args.eval_after,
            train_log=args.train_log,
            samples_per_class=args.samples_per_class,
            dataset_name=args.dataset_name,
            num_labels=args.num_labels,
            out_file=args.out_file,
            config={
                "lambda_cot_max": args.lambda_cot_max,
                "lambda_diff_max": args.lambda_diff_max,
                "warm_up": args.warm_up,
                "batch_size": args.batch_size,
                "posterior_threshold": args.posterior_threshold,
                "dataloader_wrapper_instance": data_loaders,
            },
        )
    else:
        print("Exiting without training")
    ## test model performance on unseen test set
    # eval_model(model=model, test_dataloader=data_loaders.test_dataloader, device=device)
