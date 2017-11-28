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

# import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

TOP_DIR = 'aclImdb'
use_cuda = torch.cuda.is_available()
# use_cuda = False
UNK_W = '$'
UNK_C = '#'

def shuffle2(a,b):
    assert len(a) == len(b)
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)

class MixedNet(nn.Module):
    def __init__(self, **kwargs):
        print kwargs
        self.max_sent_len = kwargs['max_sent_len']
        # self.max_word_len = kwargs['max_word_len']
        self.use_dropout = kwargs['dropout']
        nn.Module.__init__(self)

        self.word_to_ix = {}
        self.V_wrd = 0

        self.char_to_ix = {}
        self.V_chr = 0

        self.setVocab()    # Set the above 4 parameters

        self.d_wrd = kwargs['word_embed_size']
        self.W_wrd = nn.DataParallel(nn.Embedding(self.V_wrd, self.d_wrd))
        if use_cuda: self.W_wrd = self.W_wrd.cuda()
        self.k_wrd = kwargs['word_context']
        assert self.k_wrd%2 == 1 and self.k_wrd >= 3
        self.clu1 = kwargs['word_conv_units']

        self.d_chr = kwargs['char_embed_size']
        self.W_chr = nn.DataParallel(nn.Embedding(self.V_chr, self.d_chr))
        if use_cuda: self.W_chr = self.W_chr.cuda()
        self.k_chr = kwargs['char_context']
        assert self.k_chr%2 == 1 and self.k_chr >= 3
        self.clu0 = kwargs['char_conv_units']

        self.hlu = kwargs['hidden_units']
        self.batch_size = kwargs['batch_size']

        self.word_level_conv = nn.DataParallel(
            nn.Conv2d(
                1, self.clu1,
                kernel_size=(self.d_wrd + self.clu0, self.k_wrd),
                padding=(0,(self.k_wrd-1)/2)
            )
        )
        self.word_level_maxpool = nn.DataParallel(nn.AdaptiveMaxPool2d((self.clu1, 1)))

        self.char_level_conv = nn.DataParallel(
            nn.Conv2d(
                1, self.clu0,
                kernel_size=(self.d_chr, self.k_chr),
                padding=(0,(self.k_chr-1)/2)
            )
        )
        self.char_level_maxpool = nn.DataParallel(nn.AdaptiveMaxPool2d((self.clu0, 1)))

        self.fc1 = nn.DataParallel(nn.Linear(self.clu1, self.hlu))
        self.num_classes = 2
        self.fc2 = nn.DataParallel(nn.Linear(self.hlu, self.num_classes))
        if self.use_dropout:
            self.dropout = nn.DataParallel(nn.Dropout(0.5))

        # self.regex = re.compile(r'[\s,."\':;\(\)\{\}\[\]]+')

    def setVocab(self):
        with open(os.path.join(TOP_DIR, 'imdb.vocab')) as vocab_file:
            words = [word.strip() for word in vocab_file.readlines()]
            word_dict = {}
            char_set = set()
            for index, w in enumerate(words):
                word_dict[w] = index
                for c in w:
                    char_set.add(c)

            word_dict[UNK_W] = len(word_dict)
            self.word_to_ix = word_dict
            self.V_wrd = len(word_dict)

            char_dict = {}
            for ind, char in enumerate(char_set):
                char_dict[char] = ind

            char_dict[UNK_W] = len(char_dict)
            char_dict[UNK_C] = len(char_dict)
            self.char_to_ix = char_dict
            self.V_chr = len(char_dict)

    def forward(self, list_of_sentences):
        assert len(list_of_sentences) == self.batch_size
        max_sentence_length = args.max_sent_len
        # max_word_length = args.max_word_len
        max_word_length = 0
        for index, sentence in enumerate(list_of_sentences):
            # sentence = sentence.lower()
            # words = re.split(self.regex, sentence)
            words = sentence.split(' ')
            words = [s if s in self.word_to_ix else UNK_W for s in words]

            # max_sentence_length = max(max_sentence_length, len(words))
            max_word_length = max(max_word_length, max((len(x) for x in words)))

            try: assert len(words) == max_sentence_length
            except: print len(words); raise AssertionError
            list_of_sentences[index] = words

        for list_of_words in list_of_sentences:
            # list_of_words.extend([UNK_W] * (max_sentence_length - len(list_of_words)) )

            for ind, word in enumerate(list_of_words):
                list_of_words[ind] = word + (UNK_C * (max_word_length - len(word)))
                assert len(list_of_words[ind]) == max_word_length

        char_vecs = var(torch.zeros(self.batch_size, self.clu0, max_sentence_length))
        if use_cuda: char_vecs = char_vecs.cuda()
        for i, old_list_of_words in enumerate(list_of_sentences):
            list_of_words = [word for word in old_list_of_words]
            for index, word in enumerate(list_of_words):
                list_of_words[index] = [self.char_to_ix[char] for char in word]
                assert len(list_of_words[index]) == max_word_length

            char_var = var(torch.LongTensor(list_of_words))
            if use_cuda: char_var = char_var.cuda()

            char_var = self.W_chr(char_var).transpose(1,2)
            assert char_var.size() == (max_sentence_length, self.d_chr, max_word_length)
            char_var = char_var.contiguous()

            char_var = F.relu(self.char_level_conv(char_var.view(max_sentence_length, 1, self.d_chr, max_word_length)))
            assert char_var.size() == (max_sentence_length, self.clu0, 1, max_word_length)

            char_var = self.char_level_maxpool(char_var.view(max_sentence_length, self.clu0, max_word_length))
            assert char_var.size() == (max_sentence_length, self.clu0, 1)
            char_var = char_var.view(max_sentence_length, self.clu0).transpose(0,1)
            assert char_var.size() == (self.clu0, max_sentence_length)
            char_var = char_var.contiguous()
            char_vecs[i] = char_var
            # del char_var
            # gc.collect()

        for index, one_list in enumerate(list_of_sentences):
            list_of_sentences[index] = [self.word_to_ix[word.rstrip(UNK_C)] for word in one_list]
            assert len(list_of_sentences[index]) == max_sentence_length
        assert len(list_of_sentences) == self.batch_size

        word_var = var(torch.LongTensor(list_of_sentences))
        if use_cuda: word_var = word_var.cuda()

        word_var = self.W_wrd(word_var).transpose(1,2)
        assert word_var.size() == (self.batch_size, self.d_wrd, max_sentence_length)
        word_var = word_var.contiguous()

        joint_rep = torch.cat((word_var, char_vecs), 1)
        assert joint_rep.size() == (self.batch_size, self.d_wrd + self.clu0, max_sentence_length)

        joint_rep = F.relu(self.word_level_conv(joint_rep.view(self.batch_size, 1, self.d_wrd + self.clu0, max_sentence_length)))
        assert joint_rep.size() == (self.batch_size, self.clu1, 1, max_sentence_length)

        joint_rep = self.word_level_maxpool(joint_rep.view(self.batch_size, self.clu1, max_sentence_length))
        assert joint_rep.size() == (self.batch_size, self.clu1, 1)

        joint_rep = F.relu(self.fc1(joint_rep.view(self.batch_size, self.clu1)))
        output = self.fc2(joint_rep)
        # del joint_rep
        # gc.collect()

        assert output.size() == (self.batch_size, self.num_classes)
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dwrd", "--word_embed_size", type=int, help="Word embedding size", default=30)
    parser.add_argument("-kwrd", "--word_context", type=int, help="Size of word context window", default=15)
    parser.add_argument("-clu1", "--word_conv_units", type=int, help="Number of word convolutional units", default=10)
    parser.add_argument("-dchr", "--char_embed_size", type=int, help="Character embedding size", default=15)
    parser.add_argument("-kchr", "--char_context", type=int, help="Size of character context window", default=7)
    parser.add_argument("-clu0", "--char_conv_units", type=int, help="Number of character convolutional units", default=5)
    parser.add_argument("-hlu", "--hidden_units", type=int, help="Number of hidden units", default=5)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=500)
    parser.add_argument("-d", "--dropout", help="Use dropout", action="store_true")
    parser.add_argument("--max_sent_len", type=int, help="Number of words in a sentence", default=400)
    # parser.add_argument("--max_word_len", type=int, help="Number of chars in a word", default=10)
    parser.add_argument("--num_iters", type=int, help="Number of iterations", default=5)

    args = parser.parse_args()

    with open('imdb_'+str(args.max_sent_len)+'.pickle', 'rb') as fileObj:
        (x_train, y_train, x_test, y_test) = pickle.load(fileObj)

    # Shuffle training set
    shuffle2(x_train, y_train)
    len_xtrain = len(x_train)
    len_xtest = len(x_test)

    # Build network, given hyperparameters
    net = MixedNet(
        word_embed_size=args.word_embed_size,
        word_context=args.word_context,
        word_conv_units=args.word_conv_units,
        char_embed_size=args.char_embed_size,
        char_context=args.char_context,
        char_conv_units=args.char_conv_units,
        hidden_units=args.hidden_units,
        batch_size=args.batch_size,
        dropout=args.dropout,
        max_sent_len=args.max_sent_len,
        # max_word_len=args.max_word_len
    )
    batch_size = args.batch_size
    if use_cuda: net.cuda()

    optimizer = optim.Adadelta(net.parameters())
    iterations = args.num_iters
    train_loss = []
    test_loss = []
    train_acc = []
    test_acc = []
    for epoch in xrange(iterations):
        cumulative_train = 0.0
        cumulative_test = 0.0


        correct = 0.0
        wrong = 0.0
        for index in xrange(0, len_xtrain, batch_size):
            target = var(torch.LongTensor([y_train[index:index+batch_size]]),requires_grad=False)
            target = target.view(-1)
            if use_cuda: target = target.cuda()
            class_scores = net.forward(x_train[index:index+batch_size])

            # print class_scores.size(), target.size()
            loss = F.cross_entropy(class_scores, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cumulative_train += loss.data[0]
            print "[%3d, %5d] Training Loss: %2.5f" % (epoch+1, index+1, loss.data[0])
            # del target, class_scores
            # gc.collect()
            maxval, predicted = torch.max(class_scores, 1)
            for i in xrange(len(predicted)):
                assert predicted[i].data[0] in [0,1]
                if predicted[i].data[0] == target[i].data[0]:
                    correct  += 1
                else:
                    wrong += 1
        train_acc.append(correct / (correct+wrong))

        correct, wrong = 0.0, 0.0

        for index in xrange(0, len_xtest, batch_size):
            target = var(torch.LongTensor([y_test[index:index+batch_size]]),requires_grad=False)
            target = target.view(-1)
            if use_cuda: target = target.cuda()
            class_scores = net.forward(x_test[index:index+batch_size])

            # print class_scores.size(), target.size()
            loss = F.cross_entropy(class_scores, target)
            # optimizer.zero_grad()
            loss.backward()
            # optimizer.step()

            cumulative_test += loss.data[0]
            print "[%3d, %5d] Validation Loss: %2.5f" % (epoch+1, index+1, loss.data[0])
            # del target, class_scores
            # gc.collect()
            maxval, predicted = torch.max(class_scores, 1)
            for i in xrange(len(predicted)):
                assert predicted[i].data[0] in [0,1]
                if predicted[i].data[0] == target[i].data[0]:
                    correct  += 1
                else:
                    wrong += 1
        test_acc.append(correct / (correct+wrong))

        train_loss.append(cumulative_train*args.batch_size/25000)
        test_loss.append(cumulative_test*args.batch_size/25000)
    print "Finished Training"

    with open('figs.pickle', 'wb') as fileObj:
        data = (train_loss, test_loss, train_acc, test_acc)
        pickle.dump(data, fileObj,  protocol=pickle.HIGHEST_PROTOCOL)
    plt.figure(1)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(train_loss, 'r')
    plt.plot(test_loss, 'b')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

    plt.figure(2)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.plot(train_acc, 'r')
    plt.plot(test_acc, 'b')
    plt.legend(['Training Accuracy', 'Validation Accuracy'])
    plt.show()

    correct = 0.0
    wrong = 0.0

    for index in xrange(0, len_xtrain, batch_size):
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
        # del maxval, predicted, target, class_scores
        # gc.collect()
    print "Accuracy on train: %3.5f" % (correct / (correct+wrong) ,)


    correct = 0.0
    wrong = 0.0

    for index in xrange(0, len_xtest, batch_size):
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
        # del maxval, predicted, target, class_scores
        # gc.collect()
    print "Accuracy on test: %3.5f" % (correct / (correct+wrong) ,)
