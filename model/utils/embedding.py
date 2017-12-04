import numpy as np
import torch as t
import torch.nn as nn
from torch.autograd import Variable


class Embedding(nn.Module):
    def __init__(self, path, vocab_size, embedding_size):
        super(Embedding, self).__init__()

        self.max_len = max_len
        self.embedding_size = embedding_size

        self.embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        self.embeddings.weight = nn.Parameter(t.from_numpy(np.load(path)).float(), requires_grad=False)

    def forward(self, input):
        return self.embeddings(input)
