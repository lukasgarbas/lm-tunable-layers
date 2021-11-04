import torch
from encoders import LanguageModel

class TextClassifier(torch.nn.Module):
    """
    Text Classification Head for any sentence-level classification tasks.
    This class uses representations from one (LanguageModel) or multiple language models (MultiEncoder)
    and adds additional layers to solve text classification tasks.
    """
    def __init__(
        self,
        encoder: LanguageModel,
        num_classes: int,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.encoder = encoder
        self.name = str(encoder.name) + '-classifier'
        self.encoder_embedding_length = self.encoder.embedding_length

        # linear classifier layer
        self.decoder = torch.nn.Linear(
            self.encoder_embedding_length, self.num_classes
        )

        # initialize decoder weights
        torch.nn.init.xavier_uniform_(self.decoder.weight)

        # loss function for classification
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, sentences):
        encoded_sentences = self.encoder(sentences)
        return self.decode(encoded_sentences)

    def decode(self, encoded_sentences):
        scores = self.decoder(encoded_sentences)
        return scores

    def forward_loss(self, sentences, true_labels):
        scores = self.forward(sentences)
        loss_value = self.loss_function(scores, true_labels)
        return loss_value

    def predict(self, sentences):
        with torch.no_grad():
            raw_output = self.forward(sentences)
            label_scores = torch.nn.functional.softmax(raw_output, dim=1)
            predicted_label = torch.argmax(label_scores, 1)
        return predicted_label

    @staticmethod
    def from_checkpoint(path):
        return torch.load(path)


class TextRegressor(torch.nn.Module):
    """
    Text Regression Head for any sentence-level regression tasks.
    This class uses representations from one (LanguageModel) or multiple language models (MultiEncoder)
    and adds additional layers to solve text regression tasks.
    The final linear layer is used as regressor, meaning that it maps into one dimension and does not use softmax.
    """
    def __init__(
        self,
        encoder: LanguageModel,
        num_classes: int = 1,
    ):
        super().__init__()

        self.encoder = encoder
        self.encoder_embedding_length = self.encoder.embedding_length
        self.name: str = str(encoder.name) + '-regressor'

        # linear layer with num classes = 1 for regression tasks
        self.decoder = torch.nn.Linear(
            self.encoder_embedding_length, num_classes
        )

        # initialize decoder weights
        torch.nn.init.xavier_uniform_(self.decoder.weight)

        # loss function for regression
        self.loss_function = torch.nn.MSELoss()

    def forward(self, sentences):
        encoded_sentences = self.encoder(sentences)
        return self.decode(encoded_sentences)

    def decode(self, encoded_sentences):
        logits = self.decoder(encoded_sentences).squeeze()
        return logits

    def forward_loss(self, sentences, true_labels):
        scores = self.forward(sentences)
        loss_value = self.loss_function(scores, true_labels)
        return loss_value

    def predict(self, sentences):
        with torch.no_grad():
            score = self.forward(sentences)
        return score

    @staticmethod
    def from_checkpoint(path):
        return torch.load(path)