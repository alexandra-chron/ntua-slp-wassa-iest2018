import torch
from torch import nn

from modules.nn.modules import Embed, RNNEncoder


class RNNBase:

    @staticmethod
    def hidden2vocab(output, projection):
        # output_unpacked.size() = batch_size, max_length, hidden_units
        # flat_outputs = (batch_size*max_length, hidden_units),
        # which means that it is a sequence of *all* the outputs (flattened)
        flat_output = output.contiguous().view(output.size(0) * output.size(1),
                                               output.size(2))

        # the sequence of all the output projections
        decoded_flat = projection(flat_output)

        # reshaped the flat sequence of decoded words,
        # in the original (reshaped) form (3D tensor)
        decoded = decoded_flat.view(output.size(0), output.size(1),
                                    decoded_flat.size(1))
        return decoded


class LangModel(nn.Module, RNNBase):
    def __init__(self, ntokens, **kwargs):
        super(LangModel, self).__init__()

        self.ntokens = ntokens
        self.emb_size = kwargs.get("emb_size", 100)
        self.rnn_size = kwargs.get("encoder_size", 100)
        self.rnn_layers = kwargs.get("encoder_layers", 1)
        self.rnn_dropout = kwargs.get("encoder_dropout", .0)
        self.embed_noise = kwargs.get("embed_noise", .0)
        self.embed_dropout = kwargs.get("embed_dropout", .0)
        self.tie_weights = kwargs.get("tie_weights", False)
        self.pack = kwargs.get("pack", False)

        self.embedding = Embed(ntokens, self.emb_size,
                               noise=self.embed_noise,
                               dropout=self.embed_dropout)

        self.encoder = RNNEncoder(input_size=self.emb_size,
                                  rnn_size=self.rnn_size,
                                  num_layers=self.rnn_layers,
                                  bidirectional=False,
                                  dropout=self.rnn_dropout,
                                  pack=self.pack)

        self.decoder = nn.Linear(self.rnn_size, ntokens)

        if self.tie_weights:
            self.decoder.weight = self.embedding.embedding.weight
            if self.rnn_size != self.emb_size:
                self.down = nn.Linear(self.rnn_size, self.emb_size)

    def forward(self, x, hidden=None, lengths=None):
        # embed and regularize the words
        x = self.embedding(x)

        output, hidden = self.encoder(x, hidden, lengths)

        if self.tie_weights and self.rnn_size != self.emb_size:
            output = self.down(output)

        decoded = self.hidden2vocab(output, self.decoder)

        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())

        if self.encoder.encoder.mode == 'LSTM':
            return (weight.new_zeros(self.rnn_layers, bsz, self.rnn_size),
                    weight.new_zeros(self.rnn_layers, bsz, self.rnn_size))
        else:
            return weight.new_zeros(self.rnn_layers, bsz, self.rnn_size)

