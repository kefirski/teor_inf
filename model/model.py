import torch.nn as nn

from .utils.attention.embedding import EmbeddingAttention
from .utils.encoder import Encoder
from torch.nn.utils.weight_norm import weight_norm


class Model(nn.Module):
    def __init__(self, vocab_size, n_layers, n_heads, h_size, k_size, v_size, n_lockups, n_classes, dropout=0.1):
        """
        :param n_heads: Number of attention heads
        :param h_size: hidden size of input
        :param k_size: size of projected queries and keys
        :param v_size: size of projected values
        :param dropout: drop prob
        """
        super(Model, self).__init__()

        self.vocab_size = vocab_size

        self.encoder = Encoder(n_layers, n_heads, h_size, k_size, v_size, dropout)
        self.out_attention = EmbeddingAttention(h_size, 1, dropout)

        self.fc = nn.Sequential(
            weight_norm(nn.Linear(h_size, h_size)),
            nn.SELU(),

            weight_norm(nn.Linear(h_size, 100)),
            nn.SELU(),

            weight_norm(nn.Linear(100, n_classes))
        )

    def forward(self, input):

        encoding, mask = self.encoder(input)
        out = self.out_attention(encoding, mask).squeeze(1)
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
