import re
import os
import cPickle as pickle

MASTER_DIR = "aclImdb/"
traindir = "train/"
testdir = "test/"
pos = "pos/"
neg = "neg/"

def clean(raw_html):
	cleanr = re.compile('<.*?>')
	cleantext = re.sub(cleanr, '', raw_html)
	return cleantext

def parseImdb():
	X_train = []
	Y_train = []
	X_test = []
	Y_test = []
	trainpos = os.path.join(MASTER_DIR,traindir,pos)
	fileList = os.listdir(trainpos)
	for fil in fileList:
		fd = open(os.path.join(trainpos,fil))
		X_train.append(clean(fd.readlines()[0]))
		Y_train.append(1)
	print "train pos done"
	trainneg = os.path.join(MASTER_DIR,testdir,pos)
	fileList = os.listdir(trainneg)
	for fil in fileList:
		fd = open(os.path.join(trainneg,fil))
		X_train.append(clean(fd.readlines()[0]))
		Y_train.append(0)
	print "train neg done"
	testneg = os.path.join(MASTER_DIR,testdir,neg)
	fileList = os.listdir(testneg)
	for fil in fileList:
		fd = open(testneg+fil)
		X_test.append(clean(fd.readlines()[0]))
		Y_test.append(0)
	print "test neg done"
	testpos = os.path.join(MASTER_DIR,testdir,pos)
	fileList = os.listdir(testpos)
	for fil in fileList:
		fd = open(testpos+fil)
		X_test.append(clean(fd.readlines()[0]))
		Y_test.append(1)
	print "test pos done"

	return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
	data_tuple = parseImdb()
	with open("imdb.pickle", 'wb') as fileObj:
		pickle.dump(data_tuple, fileObj, protocol=pickle.HIGHEST_PROTOCOL)
