import torch
from torch import nn
import torch.nn.functional as F
from modules.neural.attention import SelfAttention
from modules.neural.modules import Embed, RNNEncoder


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

        self.output = nn.Linear(in_features=self.encoder.feature_size*2,
                                out_features=out_size)

    def forward(self, x, lengths):
        # SOS SOS SOS SOS
        # PREPEI NA ALLAKSOUME THN FORWARD WSTE OI ALLAGES NA EINAI MONO GIA PRETR LM
        # OXI GIA SKETO WASSA

        # index of word before target word
        # In lm vocab()always idx2word[4]=[#triggerword#]
        for i, tweet in enumerate(x):
            if 4 not in tweet:
                print(i, tweet)

        idxs_cpu = [(list(x[i].cpu()).index(4)-1) for i in range(0, len(x))] # very slow
        idxs = [(x[i]==4).nonzero() for i in range(0, len(x))]
        # todo: -1 to these idxs to be correct

        embeddings = self.embedding(x)
        outputs, last_output = self.encoder(embeddings, lengths)

        # hiddens for concat
        hiddens = [outputs[i][idxs_cpu[i]] for i in range(0, len(idxs))]
        hiddens_tensor = torch.stack(hiddens)

        # concat outputs[ind] me representations

        representations, attentions = self.attention(outputs, lengths)

        new_representations = torch.cat((representations, hiddens_tensor),1)
        logits = self.output(new_representations)

        return logits, attentions


class LangModel(nn.Module):
    def __init__(self, ntokens, **kwargs):
        super(LangModel, self).__init__()

        self.ntokens = ntokens
        self.emb_size = kwargs.get("emb_size", 100)
        self.rnn_size = kwargs.get("rnn_size", 100)
        self.rnn_layers = kwargs.get("rnn_layers", 1)
        self.rnn_dropout = kwargs.get("rnn_dropout", .0)
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
                                  dropout=self.rnn_dropout)

        self.decoder = nn.Linear(self.rnn_size, ntokens)

        if self.tie_weights:
            self.decoder.weight = self.embedding.embedding.weight
            if self.rnn_size != self.emb_size:
                self.down = nn.Linear(self.rnn_size, self.emb_size)

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

    def pad_outputs(self, outputs, max_length):
        pad_length = max_length - outputs.size(1)
        outputs = F.pad(outputs, (0, 0, 0, pad_length))
        return outputs

    def forward(self, x, hidden=None, lengths=None):
        # embed and regularize the words
        x = self.embedding(x)

        output, hidden = self.encoder(x, lengths, hidden)

        if self.tie_weights and self.rnn_size != self.emb_size:
            output = self.down(output)

        # pad to initial max length
        output = self.pad_outputs(output, x.size(1))

        decoded = self.hidden2vocab(output, self.decoder)

        return decoded, hidden
