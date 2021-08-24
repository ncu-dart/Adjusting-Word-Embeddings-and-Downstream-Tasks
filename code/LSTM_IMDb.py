from __future__ import print_function
from numpy import asarray
import argparse
import matplotlib.pyplot as plt
import numpy as np
import csv
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import metrics
from tensorflow.keras.datasets import imdb
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

def load_embeddings(filename):
    embeddings_dictionary = dict()
    embedding_file = open(filename, encoding = 'utf8')
    
    for line in embedding_file:
        values = line.split()
        word = values[0]
        vector_dimensions = np.asarray(values[1:], dtype='float32')
        embeddings_dictionary[word] = vector_dimensions
    embedding_file.close()

    return embeddings_dictionary

def print_result(precision_list, recall_list, f1_list, roc_list, loss_list, acc_list, outFileName):
    print("Writing down the result in", outFileName)
    outFile = open(outFileName, 'w', encoding='utf-8')
    
    outFile.write('precision'+' ')
    for results in precision_list:
        outFile.write('%.4f' %(results)+' ')
    outFile.write('\n')    
    outFile.write('recall'+' ')
    for results in recall_list:
        outFile.write('%.4f' %(results)+' ')
    outFile.write('\n')
    outFile.write('f1_score'+' ')
    for results in f1_list:
        outFile.write('%.4f' %(results)+' ')
    outFile.write('\n')
    outFile.write('ROC'+' ')
    for results in roc_list:
        outFile.write('%.4f' %(results)+' ')
    outFile.write('\n')
    outFile.write('loss'+' ')
    for results in loss_list:
        outFile.write('%.4f' %(results)+' ')
    outFile.write('\n')
    outFile.write('accuracy'+' ')
    for results in acc_list:
        outFile.write('%.4f' %(results)+' ')
    outFile.write('\n')  
    
    outFile.close()

parser = argparse.ArgumentParser()

parser.add_argument("-en", "--emb_filename", required=True, type=str, help="filename of embeddings")
parser.add_argument("-op", "--output_path", required=True, type=str, help="filepath for saving the result")
parser.add_argument("--embedding_trainable", type=bool, default=False, help="finetine word embeddings when training")

args = parser.parse_args()

maxlen = 100

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X_train = preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

word_to_index = imdb.get_word_index()
vocab_size = len(word_to_index)+1
print('Vocab size : ', vocab_size)


words_freq_list = []
for (k,v) in imdb.get_word_index().items():
    words_freq_list.append((k,v))

sorted_list = sorted(words_freq_list, key=lambda x: x[1])


embedding_file = args.emb_filename

# Word from this index are valid words. i.e  3 -> 'the' which is the
# most frequent word
INDEX_FROM = 3

word_to_index = {k:(v+INDEX_FROM-1) for k,v in imdb.get_word_index().items()}
word_to_index["<PAD>"] = 0
word_to_index["<START>"] = 1
word_to_index["<UNK>"] = 2

embeddings_dictionary = load_embeddings(embedding_file)
# create a weight matrix for words in training docs
embedding_matrix = np.zeros((vocab_size+INDEX_FROM, 300))
# unknown words are mapped to zero vector
embedding_matrix[0] = np.array(300*[0])
embedding_matrix[1] = np.array(300*[0])
embedding_matrix[2] = np.array(300*[0])

for word, i in word_to_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

        
# Build and train model
precision_list = []        
recall_list = []
f1_list = []
roc_list = []
loss_list = []
acc_list = []

for i in range(2):
    print('Build model...')

    model = Sequential()
    model.add(layers.Embedding(vocab_size+INDEX_FROM, 300, weights=[embedding_matrix], trainable=False))
    model.add(layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    print('This is %d round.' % (i))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[metrics.BinaryAccuracy(name='binary_accuracy')])
    history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))

    yhat_probs = model.predict(X_test, verbose=1)
    # predict crisp classes for test set
    #yhat_classes = model.predict_classes(X_test, verbose=1)
    yhat_classes = np.argmax(model.predict(X_test), axis=1)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    #yhat_classes = yhat_classes[:, 0]

    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, yhat_classes)
    # precision tp / (tp + fp)
    precision = precision_score(y_test, yhat_classes,average= 'weighted')
    precision_list.append(precision)
    # recall: tp / (tp + fn)
    recall = recall_score(y_test, yhat_classes,average= 'weighted')
    recall_list.append(recall)
    # f1: 2 tp / (2 tp + fp + fn)
    f1 = f1_score(y_test, yhat_classes,average= 'weighted')
    f1_list.append(f1)
    # ROC AUC
    auc = roc_auc_score(y_test, yhat_probs)
    roc_list.append(auc)

    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    loss_list.append(loss)
    acc_list.append(acc)

print('Precision: ', precision_list)
print('Recall: ', recall_list)
print('F1 score: ', f1_list)
print('ROC AUC: ', roc_list)
print('Test loss: ', loss_list)
print('Test accuracy: ', acc_list)
    
print_result(precision_list, recall_list, f1_list, roc_list, loss_list, acc_list, args.output_path)
