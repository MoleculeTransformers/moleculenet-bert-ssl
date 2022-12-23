from transformers import BertModel, BertConfig
from transformers import BertForSequenceClassification
import torch


class BERTClassifier:
    def __init__(self, num_labels=2):
        self.configuration = BertConfig()
        self.num_labels = num_labels

    def get_model(self, model_name_or_path="shahrukhx01/smole-bert"):
        """
        Initialize pretrained bert model from huggingface model hub
        """
        # initializing a model from the bert-base-uncased style configuration
        model = BertModel(self.configuration)

        model = BertForSequenceClassification.from_pretrained(
            model_name_or_path, num_labels=self.num_labels
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        return model
