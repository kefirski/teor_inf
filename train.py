import argparse

import torch as t
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.optim import Adam

from model.model import Model
from model.utils.positional_embedding import PositionalEmbedding
from model.utils.scheduled_optim import ScheduledOptim
from utils.dataloader import Dataloader

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='inf')
    parser.add_argument('--num-iterations', type=int, default=250_000, metavar='NI',
                        help='num iterations (default: 250_000)')
    parser.add_argument('--steps', type=int, default=15, metavar='S',
                        help='num steps before optimization step (default: 80)')
    parser.add_argument('--batch-size', type=int, default=80, metavar='BS',
                        help='batch size (default: 80)')
    parser.add_argument('--num-threads', type=int, default=4, metavar='BS',
                        help='num threads (default: 4)')
    parser.add_argument('--use-cuda', type=bool, default=False, metavar='CUDA',
                        help='use cuda (default: False)')
    parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                        help='learning rate (default: 0.0005)')
    parser.add_argument('--dropout', type=float, default=0.15, metavar='D',
                        help='dropout rate (default: 0.15)')
    parser.add_argument('--save', type=str, default='trained_model', metavar='TS',
                        help='path where save trained model to (default: "trained_model")')
    parser.add_argument('--tensorboard', type=str, default='default_tb', metavar='TB',
                        help='Name for tensorboard model')
    args = parser.parse_args()

    writer = SummaryWriter(args.tensorboard)

    t.set_num_threads(args.num_threads)
    loader = Dataloader('~/projects/teor_inf/utils/data/', '~/projects/wiki.ru.bin')

    model = Model(loader.vocab_size, 3, 5, 300, 75, 100, 40, n_classes=len(loader.idx_to_label), dropout=args.dropout)
    embeddings = PositionalEmbedding(loader.preprocessed_embeddings, loader.vocab_size, 1100, 300)
    if args.use_cuda:
        model = model.cuda()

    optimizer = ScheduledOptim(
        Adam(
            [{'params': model.non_fine_parameters(), 'lr': 1e-3, 'fine': False},
             {'params': model.encoder.fine_parameters(), 'lr': 1e-3, 'fine': True}],
            betas=(0.9, 0.98),
            eps=1e-9),
        300, 3, 4000)

    crit = nn.CrossEntropyLoss()

    print('Model have initialized')

    for i in range(args.num_iterations):
        optimizer.zero_grad()

        for step in range(args.steps):
            input, target = loader.torch(args.batch_size, 'train', volatile=False)
            loss = model.loss(input, target, embeddings, crit, args.use_cuda, eval=False)
            loss.backward()

        optimizer.update_learning_rate()
        optimizer.step()

        if i % 25 == 0:
            input, target = loader.torch(args.batch_size, 'valid', volatile=True)
            loss = model.loss(input, target, embeddings, crit, args.use_cuda, eval=True)
            loss = loss.cpu().data

            writer.add_scalar('nll', loss, i)

            print('i {}, nll {}'.format(i, loss.numpy()))
            print('_________')

    t.save(model.cpu().state_dict(), args.save)
