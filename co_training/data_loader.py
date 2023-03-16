from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from data import MoleculeData
import pandas as pd
import numpy as np
import random


class MoleculeDataLoader:
    def __init__(
        self,
        dataset_name,
        batch_size=8,
        debug=0,
        train_path=None,
        val_path=None,
        test_path=None,
    ):
        self.molecule_data = MoleculeData(dataset_name, debug=debug, train_path=train_path,
            val_path=val_path,
            test_path=test_path,)
        self.batch_size = batch_size

    def create_semi_supervised_loaders(self, samples_per_class=100, n_augmentations=0):
        """
        Create Torch dataloaders for data splits
        """
        self.molecule_data.text_to_tensors()
        print("creating dataloaders")
        self.data_view_1 = None
        print(f"SAMPLES per class: {samples_per_class}")
        indices = []
        label_df = pd.DataFrame(self.molecule_data.train_labels, columns=["labels"])
        if samples_per_class > 0:
            tp = label_df[label_df["labels"] == 1].sample(
                min(
                    samples_per_class,
                    len(list(label_df[label_df["labels"] == 1].index)),
                )
            )
            tn = label_df[label_df["labels"] == 0].sample(
                min(
                    samples_per_class,
                    len(list(label_df[label_df["labels"] == 0].index)),
                )
            )
            indices = list(tp.index) + list(tn.index)
            self.data_view_1 = TensorDataset(
                self.molecule_data.train_inputs_view1[indices],
                self.molecule_data.train_masks_view1[indices],
                self.molecule_data.train_labels[indices],
            )
            self.data_view_2 = TensorDataset(
                self.molecule_data.train_inputs_view2[indices],
                self.molecule_data.train_masks_view2[indices],
                self.molecule_data.train_labels[indices],
            )

        else:
            self.data_view_1 = TensorDataset(
                self.molecule_data.train_inputs_view1,
                self.molecule_data.train_masks_view1,
                self.molecule_data.train_labels,
            )
            self.data_view_2 = TensorDataset(
                self.molecule_data.train_inputs_view2,
                self.molecule_data.train_masks_view2,
                self.molecule_data.train_labels,
            )
        labelled_sampler_view1 = RandomSampler(self.data_view_1)
        self.view_1_dataloader = DataLoader(
            self.data_view_1, sampler=labelled_sampler_view1, batch_size=self.batch_size
        )

        labelled_sampler_view2 = RandomSampler(self.data_view_2)
        self.view_2_dataloader = DataLoader(
            self.data_view_2, sampler=labelled_sampler_view2, batch_size=self.batch_size
        )

        ## create unlabelled dataloader
        unlabelled_indices = list(label_df.drop(indices, axis=0).index)
        self.unlabelled_data_view1 = [
            {
                "train_inputs": self.molecule_data.train_inputs_view1[unlab_index],
                "train_masks": self.molecule_data.train_masks_view1[unlab_index],
                "train_labels": self.molecule_data.train_labels[unlab_index],
            }
            for unlab_index in unlabelled_indices
        ]

        self.unlabelled_data_view2 = [
            {
                "train_inputs": self.molecule_data.train_inputs_view2[unlab_index],
                "train_masks": self.molecule_data.train_masks_view2[unlab_index],
                "train_labels": self.molecule_data.train_labels[unlab_index],
            }
            for unlab_index in unlabelled_indices
        ]

        print(
            f"total data {len(label_df)} \
            data_view 1 {len(self.data_view_1)} \
            data_view 2 {len(self.data_view_2)} \
            unlabelled data view1 {len(self.unlabelled_data_view1)} \
            unlabelled data view2 {len(self.unlabelled_data_view2)}"
        )

        validation_data_view1 = TensorDataset(
            self.molecule_data.validation_inputs_view1,
            self.molecule_data.validation_masks_view1,
            self.molecule_data.validation_labels,
        )
        validation_sampler_view1 = SequentialSampler(validation_data_view1)
        self.validation_dataloader_view1 = DataLoader(
            validation_data_view1,
            sampler=validation_sampler_view1,
            batch_size=max(self.batch_size, len(validation_data_view1)),
        )

        validation_data_view2 = TensorDataset(
            self.molecule_data.validation_inputs_view2,
            self.molecule_data.validation_masks_view2,
            self.molecule_data.validation_labels,
        )
        validation_sampler_view2 = SequentialSampler(validation_data_view2)
        self.validation_dataloader_view2 = DataLoader(
            validation_data_view2,
            sampler=validation_sampler_view2,
            batch_size=max(self.batch_size, len(validation_data_view1)),
        )

        test_data_view1 = TensorDataset(
            self.molecule_data.test_inputs_view1,
            self.molecule_data.test_masks_view1,
            self.molecule_data.test_labels,
        )
        test_sampler_view1 = SequentialSampler(test_data_view1)
        self.test_dataloader_view1 = DataLoader(
            test_data_view1,
            sampler=test_sampler_view1,
            batch_size=min(self.batch_size, len(test_data_view1)),
        )
        test_data_view2 = TensorDataset(
            self.molecule_data.test_inputs_view2,
            self.molecule_data.test_masks_view2,
            self.molecule_data.test_labels,
        )
        test_sampler_view2 = SequentialSampler(test_data_view2)
        self.test_dataloader_view2 = DataLoader(
            test_data_view2,
            sampler=test_sampler_view2,
            batch_size=min(self.batch_size, len(test_data_view2)),
        )
        print("finished creating dataloaders")


if __name__ == "__main__":
    spam_loader = MoleculeDataLoader(dataset_name="bace")
    spam_loader.create_loaders()
