import torch as t
import torch.nn as nn
from torch.nn.utils.weight_norm import weight_norm

from .utils.attention.embedding import EmbeddingAttention
from .utils.encoder import Encoder
from .utils.resnet import ResNet


class Model(nn.Module):
    def __init__(self, vocab_size, h_size, n_lockups, n_classes, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(Model, self).__init__()

        self.vocab_size = vocab_size

        self.out_attention = EmbeddingAttention(h_size, n_lockups, dropout)

        self.conv = nn.Sequential(
            weight_norm(nn.Conv1d(n_lockups, 20, 3, 1, 1, bias=False)),
            nn.SELU(),

            nn.Dropout(dropout),

            weight_norm(nn.Conv1d(20, 10, 3, 1, 1, bias=False)),
            nn.SELU(),

            nn.Dropout(dropout),

            ResNet(10, 3),

            weight_norm(nn.Conv1d(10, 1, 3, 1, 1, bias=False)),
            nn.SELU()
        )

        self.fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, 100)),
            nn.SELU(),

            nn.Dropout(dropout),

            weight_norm(nn.Linear(100, n_classes))
        )

    def forward(self, input):

        mask = t.eq(input.abs().sum(2), 0).data

        out = self.out_attention(input, mask)
        out = self.conv(out).squeeze(1)
        return self.fc(out)

    def loss(self, input, target, embeddings, crit, cuda, eval=False):

        if eval:
            self.eval()
        else:
            self.train()

        input = embeddings(input)

        if cuda:
            input, target = input.cuda(), target.cuda()

        logits = self(input)

        return crit(logits, target)
