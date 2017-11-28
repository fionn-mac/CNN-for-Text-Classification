import os
import re
import numpy as np

import torch
from torch.autograd import Variable as var
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cPickle as pickle
import argparse

TOP_DIR = 'aclImdb'
use_cuda = torch.cuda.is_available()
UNK = 'unk'
# use_cuda = False

def shuffle2(a,b):
    assert len(a) == len(b)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

class MixedNet(nn.Module):
    def __init__(self, **kwargs):
        print kwargs
        nn.Module.__init__(self)
        self.use_dropout = kwargs['dropout']
        self.batch_size = kwargs['batch_size']
        self.d_wrd = kwargs['word_embed_size']
        self.k_wrd = kwargs['word_context']
        self.clu1 = kwargs['word_conv_units']
        self.hlu = kwargs['hidden_units']
        self.drop_amount = kwargs['drop_amount']

        self.word_to_ix = {}
        self.V_wrd = 0

        self.setVocab()    # Set the above 4 parameters

        self.W_wrd = nn.DataParallel(nn.Embedding(self.V_wrd, self.d_wrd))
        if use_cuda: self.W_wrd = self.W_wrd.cuda()
        assert self.k_wrd%2 == 1 and self.k_wrd >= 3

        self.word_level_conv = nn.DataParallel(nn.Conv2d(
            1, self.clu1,
            kernel_size=(self.d_wrd, self.k_wrd),
            padding=(0,(self.k_wrd-1)/2)
        ))
        self.word_level_maxpool = nn.DataParallel(nn.AdaptiveMaxPool2d((self.clu1, 1)))

        self.num_classes = 2
        if self.hlu == 0:
            self.fc1 = nn.DataParallel(nn.Linear(self.clu1, self.num_classes))
        else:
            self.fc1 = nn.DataParallel(nn.Linear(self.clu1, self.hlu))
            self.fc2 = nn.DataParallel(nn.Linear(self.hlu, self.num_classes))

        if self.use_dropout:
            self.dropout = nn.DataParallel(nn.Dropout(self.drop_amount))

    def setVocab(self):
        with open(os.path.join(TOP_DIR, 'imdb.vocab')) as vocab_file:
            words = [word.strip() for word in vocab_file.readlines()]
            word_dict = {}
            char_set = set()
            for index, w in enumerate(words):
                word_dict[w] = index
                for c in w:
                    char_set.add(c)
            self.word_to_ix = word_dict
            self.V_wrd = len(word_dict)

    def forward(self, word_index_list):
        assert len(word_index_list) == self.batch_size

        lookup_block = torch.LongTensor(word_index_list)
        lookup_block = var(lookup_block, requires_grad=False)
        if use_cuda: lookup_block = lookup_block.cuda()
        assert lookup_block.size() == (self.batch_size, max_words)

        sent_block = self.W_wrd(lookup_block).transpose(1,2)
        assert sent_block.size() == (self.batch_size, self.d_wrd, max_words)
        sent_block = sent_block.contiguous()

        input_to_conv = sent_block.view(self.batch_size, 1, self.d_wrd, max_words)
        x1 = F.relu(self.word_level_conv(input_to_conv))
        assert x1.size() == (self.batch_size, self.clu1, 1, max_words)
        y1 = x1.view(self.batch_size, self.clu1, max_words)

        y2 = self.word_level_maxpool(y1)
        assert y2.size() == (self.batch_size,self.clu1,1)

        y2 = y2.view(self.batch_size, self.clu1)
        if self.hlu == 0:
            y2 = self.fc1(y2)
            if self.use_dropout:
                y2 = self.dropout(y2)
        else:
            y2 = F.relu(self.fc1(y2))
            if self.use_dropout:
                y2 = self.dropout(y2)
            y2 = self.fc2(y2)
        assert y2.size() == (self.batch_size, self.num_classes)
        return y2

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dwrd", "--word_embed_size", type=int, help="Word embedding size", default=30)
    parser.add_argument("-kwrd", "--word_context", type=int, help="Size of word context window", default=7)
    parser.add_argument("-clu0", "--word_conv_units", type=int, help="Number of word convolutional units", default=10)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=500)
    parser.add_argument("-hlu", "--hidden_units", type=int, help="Number of hidden units", default=0)
    parser.add_argument("-d", "--dropout", help="Use dropout", action="store_true")
    parser.add_argument("--drop_amount", help="How much dropout to use", type=float, default=0.5)
    parser.add_argument("--maxlen", type=int, help="Maximum number of words in a sentence", default=400)
    parser.add_argument("--num_iters", type=int, help="Number of iterations", default=5)

    args = parser.parse_args()

    with open('imdb.pickle', 'rb') as fileObj:
        (x_train, y_train, x_test, y_test) = pickle.load(fileObj)

    # Shuffle training set
    shuffle2(x_train, y_train)

    # Build network, given hyperparameters
    net = MixedNet(
        word_embed_size=args.word_embed_size,
        word_context=args.word_context,
        word_conv_units=args.word_conv_units,
        batch_size=args.batch_size,
        hidden_units=args.hidden_units,
        dropout=args.dropout,
        drop_amount=args.drop_amount
    )
    batch_size = args.batch_size
    if use_cuda: net.cuda()

    regex = re.compile(r'[\s,."\':;\(\)\{\}\[\]]+')
    max_words = args.maxlen

    for index, sentence in enumerate(x_train):
        sentence = sentence.lower()
        words = re.split(regex, sentence)
        words = [s if s in net.word_to_ix else UNK for s in words[:max_words]]
        words.extend([UNK] * (max_words - len(words)))
        x_train[index] = [net.word_to_ix[word] for word in words]
        assert len(x_train[index]) == max_words
    print "train set done"

    for index, sentence in enumerate(x_test):
        sentence = sentence.lower()
        words = re.split(regex, sentence)
        words = [s if s in net.word_to_ix else UNK for s in words[:max_words]]
        words.extend([UNK] * (max_words - len(words)))
        x_test[index] = [net.word_to_ix[word] for word in words]
        assert len(x_test[index]) == max_words
    print "test set done"

    print net.parameters
    optimizer = optim.Adam(net.parameters())
    iterations = 5
    for epoch in xrange(args.num_iters):
        for index in xrange(0, len(x_train), batch_size):
            target = var(torch.LongTensor([y_train[index:index+batch_size]]),requires_grad=False)
            target = target.view(-1)
            if use_cuda: target = target.cuda()
            class_scores = net.forward(x_train[index:index+batch_size])

            # print class_scores.size(), target.size()
            loss = F.cross_entropy(class_scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print "[%3d, %5d] loss: %2.5f" % (epoch+1, index+1, loss.data[0])

    print "Finished training"
    correct = 0.0
    wrong = 0.0

    for index in xrange(0, len(x_train), batch_size):
        target = y_train[index:index+batch_size]
        class_scores = net.forward(x_train[index:index+batch_size])

        maxval, predicted = torch.max(class_scores, 1)
        # print class_scores.size(), target.size()

        for i in xrange(len(predicted)):
            assert predicted[i].data[0] in [0,1]
            if predicted[i].data[0] == target[i]:
                correct  += 1
            else:
                wrong += 1
    print "Accuracy on train: %2.5f" % (correct / (correct+wrong) ,)

    for index in xrange(0, len(x_test), batch_size):
        target = y_test[index:index+batch_size]
        class_scores = net.forward(x_test[index:index+batch_size])

        maxval, predicted = torch.max(class_scores, 1)
        # print class_scores.size(), target.size()

        for i in xrange(len(predicted)):
            assert predicted[i].data[0] in [0,1]
            if predicted[i].data[0] == target[i]:
                correct  += 1
            else:
                wrong += 1
    print "Accuracy on test: %2.5f" % (correct / (correct+wrong) ,)
