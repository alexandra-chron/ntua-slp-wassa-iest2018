from torch import nn, torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from modules.nn.regularization import GaussianNoise



class RecurrentHelper:
    @staticmethod
    def last_by_index(outputs, lengths):
        # Index of the last output for each sequence.
        idx = (lengths - 1).view(-1, 1).expand(outputs.size(0),
                                               outputs.size(2)).unsqueeze(1)
        return outputs.gather(1, idx).squeeze()

    def last_timestep(self, outputs, lengths, bi=False):
        if bi:
            forward, backward = self.split_directions(outputs)
            last_forward = self.last_by_index(forward, lengths)
            last_backward = backward[:, 0, :]
            return torch.cat((last_forward, last_backward), dim=-1)

        else:
            return self.last_by_index(outputs, lengths)

    @staticmethod
    def split_directions(outputs):
        direction_size = int(outputs.size(-1) / 2)
        forward = outputs[:, :, :direction_size]
        backward = outputs[:, :, direction_size:]
        return forward, backward

    def pad_outputs(self, out_packed, max_length):

        out_unpacked, _lengths = pad_packed_sequence(out_packed,
                                                     batch_first=True)

        # pad to initial max length
        pad_length = max_length - out_unpacked.size(1)
        out_unpacked = F.pad(out_unpacked, (0, 0, 0, pad_length))
        return out_unpacked

    @staticmethod
    def sort_by(lengths):
        """
        Sort batch data and labels by length.
        Useful for variable length inputs, for utilizing PackedSequences
        Args:
            lengths (nn.Tensor): tensor containing the lengths for the data

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

        if lengths.data.is_cuda:
            reverse_idx = reverse_idx.cuda()

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


class RNNEncoder(nn.Module, RecurrentHelper):
    def __init__(self, input_size,
                 rnn_size,
                 num_layers=1,
                 bidirectional=False,
                 dropout=0.,
                 pack=False):
        """
        A simple RNN Encoder, which produces a fixed vector representation
        for a variable length sequence of feature vectors, using the output
        at the last timestep of the RNN.
        Args:
            input_size (int): the size of the input features
            rnn_size (int):
            num_layers (int):
            bidirectional (bool):
            dropout (float):
        """
        super(RNNEncoder, self).__init__()

        self.pack = pack

        self.encoder = nn.LSTM(input_size=input_size,
                               hidden_size=rnn_size,
                               num_layers=num_layers,
                               bidirectional=bidirectional,
                               dropout=dropout,
                               batch_first=True)

        # the dropout "layer" for the output of the RNN
        self.drop_rnn = nn.Dropout(dropout)

        # define output feature size
        self.feature_size = rnn_size

        # double if bidirectional
        if bidirectional:
            self.feature_size *= 2

    def forward(self, x, hidden=None, lengths=None):

        batch, max_length, feat_size = x.size()
        if lengths is not None and self.pack:
            packed = pack_padded_sequence(x, list(lengths.data),
                                          batch_first=True)

            out_packed, hidden = self.encoder(packed)

            out_unpacked = self.pad_outputs(out_packed, max_length)

            outputs = self.drop_rnn(out_unpacked)
        else:
            outputs, hidden = self.encoder(x, hidden)
        return outputs, hidden

class Embed(nn.Module):
    def __init__(self,
                 num_embeddings,
                 embedding_dim,
                 embeddings=None,
                 noise=.0,
                 dropout=.0,
                 trainable=False):
        """
        Define the layer of the model and perform the initializations
        of the layers (wherever it is necessary)
        Args:
            embeddings (numpy.ndarray): the 2D ndarray with the word vectors
            noise (float):
            dropout (float):
            trainable (bool):
        """
        super(Embed, self).__init__()

        # define the embedding layer, with the corresponding dimensions
        self.embedding = nn.Embedding(num_embeddings=num_embeddings,
                                      embedding_dim=embedding_dim)

        if embeddings is not None:
            print("Initializing Embedding layer with pre-trained weights!")
            self.init_embeddings(embeddings, trainable)

        # the dropout "layer" for the word embeddings
        self.dropout = nn.Dropout(dropout)

        # the gaussian noise "layer" for the word embeddings
        self.noise = GaussianNoise(noise)

    def init_embeddings(self, weights, trainable):
        self.embedding.weight = nn.Parameter(torch.from_numpy(weights),
                                             requires_grad=trainable)

    def forward(self, x):
        """
        This is the heart of the model. This function, defines how the data
        passes through the network.
        Args:
            x (): the input data (the sentences)

        Returns: the logits for each class

        """
        embeddings = self.embedding(x)

        if self.noise.stddev > 0:
            embeddings = self.noise(embeddings)

        if self.dropout.p > 0:
            embeddings = self.dropout(embeddings)

        return embeddings
