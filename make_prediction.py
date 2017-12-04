import argparse

import numpy as np
import torch as t
import torch.nn.functional as F

from model.model import Model
from model.utils.positional_embedding import PositionalEmbedding
from utils.dataloader import Dataloader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='inf')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    args = parser.parse_args()

    t.set_num_threads(args.num_threads)
    loader = Dataloader('~/projects/teor_inf/utils/data/', '~/projects/wiki.ru.bin')

    model = Model(loader.vocab_size, 4, 10, 300, 30, 30, 9, n_classes=len(loader.idx_to_label))
    embeddings = PositionalEmbedding(loader.preprocessed_embeddings, loader.vocab_size, 1100, 300)

    model.load_state_dict(t.load(args.save))

    if args.use_cuda:
        model = model.cuda()

    model.eval()

    result = []

    for input in loader.test_data():

        input = embeddings(input)
        if args.use_cuda:
            input = input.cuda()

        logits = model(input)
        prediction = F.softmax(logits, dim=1)
        prediction = prediction.cpu().data.numpy()

        result += [loader.idx_to_label[np.argmax(p)] for p in prediction]

    text_file = open("prediction.txt", "w")
    text_file.write('\n'.join(result))
    text_file.close()
