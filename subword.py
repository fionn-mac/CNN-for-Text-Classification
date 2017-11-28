import numpy as np
from parse import *
import math
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import torch.autograd as autograd

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = LongTensor

MAXLEN = 500
EMBED_SIZE = 128
CONTEXT_WINDOW = 3
OPCHANNELS = 64
OPMAXPOOL = 3
BATCH_SIZE = 128
HIDDEN_DIMENSION1 = 128
HIDDEN_DIMENSION2 = 128
REC_LAYERS1 = 2
REC_LAYERS2 = 2
NUM_EPOCHS = 6
#
# MAXLEN = 500
# EMBED_SIZE = 3
# CONTEXT_WINDOW = 3
# OPCHANNELS = 1
# OPMAXPOOL = 4
# BATCH_SIZE = 1
# HIDDEN_DIMENSION1 = 2
# HIDDEN_DIMENSION2 = 2
# REC_LAYERS1 = 2
# REC_LAYERS2 = 2
# NUM_EPOCHS = 1

def shuffle2(a,b):
	assert len(a) == len(b)
	rng_state = np.random.get_state()
	np.random.shuffle(a)
	np.random.set_state(rng_state)
	np.random.shuffle(b)
	return a,b

def convert(arr, Y):
	charArr = []
	yarr = []
	i=0
	for line in arr:
		line = line.lower()
		line = line.split()
		smallList = []
		for word in line:
			for char in word:
				smallList.append(char)
		if len(smallList)<=MAXLEN:
			charArr.append(smallList)
			yarr.append(Y[i])
		i+=1

	return charArr,yarr

def toNum(train, test):
	charVocab = []
	num2char={}
	char2num={}
	for line in train:
		charVocab = set(charVocab + line)
		charVocab = list(charVocab)
	for line in test:
		charVocab = set(charVocab + line)
		charVocab = list(charVocab)

	i=0
	for ele in charVocab:
		num2char[i] = ele
		char2num[ele]= i
		i+=1

	xtrain = []
	xtest = []
	for line in train:
		smallArr = []
		for ele in line:
			smallArr.append(char2num[ele])
		xtrain.append(smallArr)

	for line in test:
		smallArr = []
		for ele in line:
			smallArr.append(char2num[ele])
		xtest.append(smallArr)

	return xtrain,xtest,num2char,char2num,i

class Network(nn.Module):
	def __init__(self, args):
		super(Network,self).__init__()
		self.embedding_size = args[0]
		self.num_embeddings = args[1]
		self.context_window = args[2]
		self.output_channels = args[3]
		self.maxPoolOp = args[4]
		self.batch_size = args[5]
		self.hidden_dimension1 = args[6]
		self.recLayers1 = args[7]
		self.hidden_dimension2 = args[8]
		self.recLayers2 = args[9]

		self.some = int(math.ceil(MAXLEN/self.maxPoolOp))
		self.char_embedding = nn.Embedding(self.num_embeddings, self.embedding_size)
		self.char_convolution = nn.Conv2d(1, self.output_channels,
			kernel_size = (self.context_window,self.embedding_size),
			stride=1,
			padding = ((self.context_window - 1)/2, 0))
		self.char_pool = nn.MaxPool1d(self.maxPoolOp)

		self.hidden1 = self.init_hidden(self.recLayers1,self.hidden_dimension1)
		self.lstm1 = nn.LSTM(self.output_channels, self.hidden_dimension1, self.recLayers1)

		self.hidden2 = self.init_hidden(self.recLayers2,self.hidden_dimension2)
		self.lstm2 = nn.LSTM(self.hidden_dimension1, self.hidden_dimension2, self.recLayers2)

		self.fc = nn.Linear(self.hidden_dimension2*self.some,2)

	def init_hidden(self,r,h):
		x = Variable(torch.randn(r,self.batch_size,h))
		y = Variable(torch.randn(r,self.batch_size,h))
		if use_cuda:
			x = x.cuda()
			y = y.cuda()
		return (x,y)

	def forward(self, sentence):

		M = MAXLEN
		batchInput = []
		for each in sentence:
			arr = []
			for ele in each:
				arr.append(ele)
			while(len(arr)<M):
				arr.append(self.num_embeddings-1)
			batchInput.append(arr)

		batchInput = Variable(LongTensor(batchInput), requires_grad = False)
		# print batchInput.size()

		embedding_of_sentence = self.char_embedding(batchInput)
		# print embedding_of_sentence.size()

		x1 = embedding_of_sentence.view(self.batch_size,1,M, self.embedding_size)
		x2 = F.relu(self.char_convolution(x1))
		x3 = self.char_pool(x2.view(self.batch_size, self.output_channels, M))

		x3 = x3.transpose(0,2).transpose(1,2)
		x3 = x3.contiguous()

		# print x3.size()
		x4, self.hidden1 = self.lstm1(x3,self.hidden1)
		x5, self.hidden2 = self.lstm2(x4,self.hidden2)
		# print x5.size()

		x5 = x5.transpose(0,1)
		x5 = x5.contiguous()

		x6 = F.softmax(self.fc(x5.view(self.batch_size,
							self.hidden_dimension2*self.some)))
		return x6

if __name__ == '__main__':
	# args = [EMBED_SIZE,8,CONTEXT_WINDOW,OPCHANNELS,OPMAXPOOL,BATCH_SIZE,
	# HIDDEN_DIMENSION1, REC_LAYERS1, HIDDEN_DIMENSION2, REC_LAYERS2]
	#
	# a = [[1,2,3],[1,2,3],[1,2,3]]
	# model = Network(args)
	# if use_cuda:
	# 	model.cuda()
	#
	# x = model.forward(a)
	# print x.size()
	#

	print "Retreiving Xtrain,Ytrain,Xtest,Ytest"
	X_train, Y_train, X_test, Y_test = parse()
	print "done"

	print "converting data to chars"
	X_train, Y_train = convert(X_train, Y_train)
	X_test, Y_test = convert(X_test, Y_test)
	print "conversion done"

	print "Length of Train", len(X_train), len(Y_train)
	print "Length of Test", len(X_test), len(Y_test)

	print "converting characters to numbers"
	X_train, X_test, num2char, char2num, charCount = toNum(X_train, X_test)
	# X_train = np.asarray(X_train)
	# X_test = np.asarray(X_test)
	# Y_train = np.asarray(Y_train).flatten()
	# Y_test = np.asarray(Y_test).flatten()
	print len(X_train),len(Y_train),len(X_test),len(Y_test)
	print "conversion done"
	# print X_train[0]
	X_train, Y_train = shuffle2(X_train,Y_train)

	args = [EMBED_SIZE,charCount+1,CONTEXT_WINDOW,OPCHANNELS,OPMAXPOOL,BATCH_SIZE,
	HIDDEN_DIMENSION1, REC_LAYERS1, HIDDEN_DIMENSION2, REC_LAYERS2]

	model = Network(args)
	if use_cuda:
		model.cuda()

	optimizer = optim.Adamax(model.parameters())
	loss_function = nn.CrossEntropyLoss()

	for i in xrange(NUM_EPOCHS):
		print "Iteration Number: ",i
		j = 0
		while(j<len(X_train) - BATCH_SIZE):
			print "Batch Starting from: ",j
			batchX = []
			batchY = []
			for k in range(j, j+BATCH_SIZE):
				batchX.append(X_train[k])
				batchY.append(Y_train[k])
			optimizer.zero_grad()
			predictedY = model(batchX)
			batchY = Variable(LongTensor(batchY), requires_grad=False)
			loss = loss_function(predictedY, batchY)
			loss.backward(retain_graph=True)
			optimizer.step()
			j+=BATCH_SIZE
		if(j>len(X_train) and j<len(X_train)):
			j = len(X_train) - BATCH_SIZE
			batchX = []
			batchY = []
			for k in range(j, j+BATCH_SIZE):
				batchX.append(X_train[k])
				batchY.append(Y_train[k])
			optimizer.zero_grad()
			predictedY = model(batchX)
			batchY = Variable(LongTensor(batchY), requires_grad=False)
			loss = loss_function(predictedY, batchY)
			loss.backward(retain_graph=True)
			optimizer.step()

	i = 0
	correct = 0
	total = 0
	while(i<len(X_test)-BATCH_SIZE):
		batchX = []
		batchY = []
		for k in range(i, i+BATCH_SIZE):
			batchX.append(X_test[k])
			batchY.append(Y_test[k])
		predictedY = model(batchX)
		for j in xrange(BATCH_SIZE):
			val, indi = predictedY[j].max(0)
			if indi == Y_test[j+i]:
				correct+=1
			total+=1
		i+=BATCH_SIZE

	print correct/total
	print "Fucking Successful"
