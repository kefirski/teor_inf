import torch.nn as nn


class SeqToSeq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, rnn=nn.GRU):
        super(SeqToSeq, self).__init__()

        self.rnn = rnn(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            bias=False
        )

    def forward(self, input):
        result, _ = self.rnn(input)
        return result
