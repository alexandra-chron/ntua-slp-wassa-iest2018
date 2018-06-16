from torch import nn, torch

from config import DEVICE
from modules.neural.attention import SelfAttention
from modules.neural.modules import Embed, RNNEncoder


class ModelHelper:
    @staticmethod
    def _sort_by(lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (neural.Tensor): tensor containing the lengths for the data

        Returns:
            - sorted lengths Tensor
            - sort (callable) which will sort a given iterable
                according to lengths
            - unsort (callable) which will revert a given iterable to its
                original order

        """
        batch_size = lengths.size(0)

        sorted_lengths, sorted_idx = lengths.sort()
        _, original_idx = sorted_idx.sort(0, descending=True)
        reverse_idx = torch.linspace(batch_size - 1, 0, batch_size).long()

        reverse_idx = reverse_idx.to(DEVICE)

        sorted_lengths = sorted_lengths[reverse_idx]

        def sort(iterable):
            if len(iterable.shape) > 1:
                return iterable[sorted_idx.data][reverse_idx]
            else:
                return iterable

        def unsort(iterable):
            if len(iterable.shape) > 1:
                return iterable[reverse_idx][original_idx][reverse_idx]
            else:
                return iterable

        return sorted_lengths, sort, unsort


class Classifier(nn.Module):
    def __init__(self, embeddings=None, out_size=1, **kwargs):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            out_size ():
        """
        super(Classifier, self).__init__()
        embed_finetune = kwargs.get("embed_finetune", False)
        embed_noise = kwargs.get("embed_noise", 0.)
        embed_dropout = kwargs.get("embed_dropout", 0.)

        encoder_size = kwargs.get("encoder_size", 128)
        encoder_layers = kwargs.get("encoder_layers", 1)
        encoder_dropout = kwargs.get("encoder_dropout", 0.)
        bidirectional = kwargs.get("encoder_bidirectional", False)
        attention_layers = kwargs.get("attention_layers", 1)
        attention_dropout = kwargs.get("attention_dropout", 0.)
        self.attention_context = kwargs.get("attention_context", False)

        ########################################################

        self.embedding = Embed(
            num_embeddings=embeddings.shape[0],
            embedding_dim=embeddings.shape[1],
            embeddings=embeddings,
            noise=embed_noise,
            dropout=embed_dropout,
            trainable=embed_finetune)

        self.encoder = RNNEncoder(input_size=embeddings.shape[1],
                                  rnn_size=encoder_size,
                                  num_layers=encoder_layers,
                                  bidirectional=bidirectional,
                                  dropout=encoder_dropout)

        self.attention = SelfAttention(self.encoder.feature_size,
                                       layers=attention_layers,
                                       dropout=attention_dropout,
                                       batch_first=True)

        self.output = nn.Linear(in_features=self.encoder.feature_size,
                                out_features=out_size)

    def forward(self, x, lengths):
        embeddings = self.embedding(x)
        outputs, last_output = self.encoder(embeddings, lengths)
        representations, attentions = self.attention(outputs, lengths)

        logits = self.output(representations)

        return logits, attentions
