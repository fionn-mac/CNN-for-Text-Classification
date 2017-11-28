from __future__ import print_function
import keras
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding, Input
from keras.layers import Conv1D, GlobalMaxPooling1D, concatenate
from keras.models import Model
from keras.datasets import imdb
from keras import regularizers
import argparse

import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dwrd", "--word_embed_size", type=int, help="Word embedding size", default=300)
    parser.add_argument("-clu0", "--word_conv_units", type=int, help="Number of word convolutional units", default=100)
    parser.add_argument("-b", "--batch_size", type=int, help="Batch Size", default=50)
    parser.add_argument("--maxlen", type=int, help="Maximum number of words in a sentence", default=400)
    parser.add_argument("--num_iters", type=int, help="Number of iterations", default=5)


    args = parser.parse_args()

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=args.maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=args.maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Building model')

    main_input = Input(shape=(args.maxlen,), dtype='int32', name='main_input')
    embeds = Embedding(output_dim=args.word_embed_size,
                        input_dim=10000, input_length=args.maxlen)(main_input)
    c1 = Conv1D(filters=args.word_conv_units,
                kernel_size=3,
                padding='same',
                activation='relu',
                kernel_initializer='glorot_normal',
                kernel_regularizer=regularizers.l2(0.1))(embeds)
    c1 = GlobalMaxPooling1D()(c1)
    c2 = Conv1D(filters=args.word_conv_units,
                kernel_size=4,
                padding='same',
                activation='relu',
                kernel_initializer='glorot_normal',
                kernel_regularizer=regularizers.l2(0.1))(embeds)
    c2 = GlobalMaxPooling1D()(c2)
    c3 = Conv1D(filters=args.word_conv_units,
                kernel_size=5,
                padding='same',
                activation='relu',
                kernel_initializer='glorot_normal',
                kernel_regularizer=regularizers.l2(0.1))(embeds)
    c3 = GlobalMaxPooling1D()(c3)

    concats = concatenate([c1,c2,c3], axis=-1)
    dropped = Dropout(0.5)(concats)
    main_output = Dense(2, activation='softmax')(dropped)

    model = Model(inputs=[main_input], outputs=[main_output])
    opt = keras.optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=opt, loss=['categorical_crossentropy'],
                    loss_weights=[1.0], metrics=['accuracy'])
    history = model.fit([x_train], [np_utils.to_categorical(y_train)],
                validation_data=(x_test, np_utils.to_categorical(y_test)),
                epochs=args.num_iters, batch_size=args.batch_size, verbose=2)

    #  "Accuracy"
    plt.figure(1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.figure(2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Training', 'Validation'], loc='upper left')
    plt.show()
