from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb
import argparse
import re
import cPickle as pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--maxlen", type=int, help="Number of words in a sentence", default=400)

    args = parser.parse_args()

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000,
                                                            index_from=3)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=args.maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=args.maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    word_to_id = imdb.get_word_index()
    word_to_id = {k:(v+3) for k,v in word_to_id.items()}
    word_to_id["$"] = 0     # padding char
    word_to_id["@"] = 1     # start char
    word_to_id["#"] = 2     # unk char

    id_to_word = {value:key for key,value in word_to_id.items()}
    exp = re.compile(r'[@#]')

    print("Filtering train set")
    xf_train = [ re.sub(exp, '$', ' '.join(id_to_word[word] for word in sentence)) for sentence in x_train]

    print("Filtering test set")
    xf_test = [ re.sub(exp, '$', ' '.join(id_to_word[word] for word in sentence)) for sentence in x_test]

    print("Dumping data to file")
    with open('imdb_'+str(args.maxlen)+'.pickle', 'wb') as fileObj:
        pickle.dump((xf_train, y_train, xf_test, y_test), fileObj,
                    protocol=pickle.HIGHEST_PROTOCOL)
    print("All done")
