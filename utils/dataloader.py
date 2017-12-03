import collections
import os
import re

import numpy as np
import pandas as pnd
import torch as t
from six.moves import cPickle
from torch.autograd import Variable

from .embeddings import EmbedFetcher


class Dataloader():
    def __init__(self, data_path='', embeddings_path='', force_preprocessing=False):
        """
        :param data_path: path to data
        :param force_preprocessing: whether to preprocess data even if it was preprocessed before
        """

        assert isinstance(data_path, str), \
            'Invalid data_path type. Required {}, but {} found'.format(str, type(data_path))

        self.data_path = data_path
        self.prep_path = self.data_path + 'preprocessings/'

        if not os.path.exists(self.prep_path):
            os.makedirs(self.prep_path)

        '''
        go_token (stop_token) uses to mark start (end) of the sequence
        pad_token uses to fill tensor to fixed-size length
        In order to make modules work correctly,
        these tokens should be unique
        '''
        self.go_token = '<GO>'
        self.pad_token = '<PAD>'
        self.stop_token = '<STOP>'

        self.pretrained_embeddings = embeddings_path

        self.data_file = {
            'train': self.data_path + 'news_train.txt',
            'test': self.data_path + 'news_test.txt'
        }

        self.idx_file = self.prep_path + 'vocab.pkl'
        self.labels_file = self.prep_path + 'labels.pkl'
        self.tensor_file = self.prep_path + 'tensor.pkl'
        self.preprocessed_embeddings = self.prep_path + 'embeddings.npy'

        idx_exists = os.path.exists(self.idx_file)
        label_exists = os.path.exists(self.labels_file)
        tensor_exists = os.path.exists(self.tensor_file)
        embed_exists = os.path.exists(self.preprocessed_embeddings)

        preprocessings_exist = all([file for file in [idx_exists, label_exists, tensor_exists, embed_exists]])

        if preprocessings_exist and not force_preprocessing:
            print('Loading preprocessed data have started')
            self.load_preprocessed()
            print('Preprocessed data have loaded')
        else:
            print('Processing have started')
            self.preprocess()
            print('Data have preprocessed')

    def build_vocab(self, sentences):
        """
        :param sentences: An array of chars in data
        :return:
            vocab_size – Number of unique words in corpus
            idx_to_word – Array of shape [vocab_size] containing list of unique chars
            word_to_idx – Dictionary of shape [vocab_size]
                such that idx_to_word[word_to_idx[some_char]] = some_char
                where some_char is is from idx_to_word
        """

        word_counts = collections.Counter(sentences)

        idx_to_word = [x[0] for x in word_counts.most_common()]
        idx_to_word = [self.pad_token, self.go_token, self.stop_token] + list(sorted(idx_to_word))

        word_to_idx = {x: i for i, x in enumerate(idx_to_word)}

        vocab_size = len(idx_to_word)

        return vocab_size, idx_to_word, word_to_idx

    @staticmethod
    def clear_line(line):
        line = re.sub(r"\/\\_-", " ", line)
        line = re.sub(r'[0-9]+', 'число', line)
        line = re.sub(r"[^a-zа-я']", ' ', line)
        line = re.sub(r"'", " '", line)
        line = re.sub(r'\s+', ' ', line)

        return line

    def preprocess(self):

        train_data = pnd.read_csv(self.data_file['train'], sep='\t', names=['class', 'title', 'article'])
        test_data = pnd.read_csv(self.data_file['test'], sep='\t', names=['title', 'article'])

        self.data = {
            'train': train_data[3000:],
            'valid': train_data[:3000],
            'test': test_data
        }
        del train_data, test_data

        self.idx_to_label = list(set(self.data['train']['class']))
        self.label_to_idx = {label: i for i, label in enumerate(self.idx_to_label)}

        for target in self.data.keys():
            self.data[target]['title'] = self.data[target]['title'].map(lambda line: self.clear_line(line.lower()))
            self.data[target]['article'] = self.data[target]['article'].map(lambda line: self.clear_line(line.lower()))

        self.data['train']['class'] = self.data['train']['class'].map(lambda label: self.label_to_idx[label])
        self.data['valid']['class'] = self.data['valid']['class'].map(lambda label: self.label_to_idx[label])

        words = [word
                 for target in self.data.keys()
                 for source in ['title', 'article']
                 for line in list(self.data[target][source])
                 for word in line.split(' ')]
        self.vocab_size, self.idx_to_word, self.word_to_idx = self.build_vocab(words)

        embeddings = EmbedFetcher.fetch(self.idx_to_word, self.pretrained_embeddings)
        np.save(self.preprocessed_embeddings, embeddings)

        for target in self.data.keys():
            for column in ['title', 'article']:
                self.data[target][column] = self.data[target][column] \
                    .map(lambda line:
                         [self.word_to_idx[self.go_token]] +
                         [self.word_to_idx[word] for word in line.split(' ')] +
                         [self.word_to_idx[self.stop_token]])

        with open(self.idx_file, 'wb') as f:
            cPickle.dump(self.idx_to_word, f)

        with open(self.labels_file, 'wb') as f:
            cPickle.dump(self.idx_to_label, f)

        with open(self.tensor_file, 'wb') as f:
            cPickle.dump(self.data, f)

    def load_preprocessed(self):

        self.idx_to_word = cPickle.load(open(self.idx_file, "rb"))
        self.vocab_size = len(self.idx_to_word)
        self.word_to_idx = dict(zip(self.idx_to_word, range(self.vocab_size)))

        self.idx_to_label = cPickle.load(open(self.labels_file, "rb"))
        self.label_to_idx = {label: i for i, label in enumerate(self.idx_to_label)}

        self.data = cPickle.load(open(self.tensor_file, "rb"))

    def next_batch(self, batch_size, target):
        """
        :param batch_size: number of selected data elements
        :return: target tensors
        """

        indexes = np.random.choice(list(self.data[target].index), size=batch_size)
        lines = self.data[target].ix[indexes]

        text = list(lines['title'])
        target = list(lines['class'])

        return self.pad_input(text), np.array(target)

    @staticmethod
    def pad_input(sequences):

        lengths = [len(line) for line in sequences]
        max_length = max(lengths)

        '''
        Pad token has idx 0 for both targets
        '''
        return np.array([line + [0] * (max_length - lengths[i])
                         for i, line in enumerate(sequences)])

    def torch(self, batch_size, target, volatile=False):

        text, target = self.next_batch(batch_size, target)
        text, target = [Variable(t.from_numpy(var), volatile=volatile)
                        for var in [text, target]]

        return text, target

    def test_data(self):

        for i in range(int(15000 / 200)):

            data = list(self.data['test'][i:i + 200]['title'])
            data = self.pad_input(data)

            yield Variable(t.from_numpy(data), volatile=True)
