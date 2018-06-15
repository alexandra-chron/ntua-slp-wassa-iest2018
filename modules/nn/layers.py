import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import Parameter, init
import numpy as np


class Attention(nn.Module):
    def __init__(self,
                 attention_size,
                 batch_first=False,
                 layers=1,
                 dropout=.0,
                 non_linearity="tanh"):
        """
        The SelfAttention layer operates on a sequence of vectors.
        The purpose of the layer is to compute fixed vector representation
        for the input sequence, by computing a weighted sum (convex combination)
        of all the vectors in the input sequence.
        Args:
            attention_size (int): The size of the vectors in the input sequence
            batch_first (bool):
            layers (int):
            dropout (float):
            non_linearity (str):
        """
        super(Attention, self).__init__()

        self.batch_first = batch_first
        self.attention_size = attention_size

        # the attention vector, has the same dimensionality as each of the
        # vectors of the input sequence
        # (otherwise we wouldn't be able to take the dot product)

        # you have two options:
        # either make it a simple Parameter, which will be updated with SGD
        # or use nn.Linear.
        # I suggest that for educational purposes you adopt the 1st approach
        # but in this case you have to initialize the Parameter yourself,
        # since otherwise it will be a zero vector and it will learn nothing.

        self.attention_weights = Parameter(torch.FloatTensor(int(self.attention_size)))
        # attention weights size = hidden size
        init.uniform(self.attention_weights.data, -0.005, 0.005)

        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def forward(self, inputs, lengths):
        """
        Args:
            inputs ():
            lengths ():
        Returns:
            representations (): the vector representation for each input sequence.
            scores (): the attention weights. Useful for interpreting
            the behaviour of our model (visualization).
        """
        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len


        # 1.1 - dot product of inputs with attention vector and get the scores
        scores = inputs.matmul(self.attention_weights)
        # representations = []

        # 1.2 - apply the activation function (non-linearity) to the scores
        scores = self.activation(scores)

        # 1.3 - softmax operation to the scores,
        # in order to normalize the distribution of weights
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # 2.1 - construct a mask, based on sentence lengths
        mask = self.get_mask(scores, lengths)
        # 2.2 - apply the mask - zero out masked timesteps
        scores = mask * scores
        # 2.3 - re-normalize the masked scores
        _sums = scores.sum(-1, keepdim=True)  # sums per row
        scores = scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # 3.1 - multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))
        # 3.2 - sum the hidden states
        representations = weighted.sum(1) #.squeeze()
        return representations, scores

