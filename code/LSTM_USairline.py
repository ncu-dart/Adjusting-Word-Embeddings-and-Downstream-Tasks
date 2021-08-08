from nltk.corpus import stopwords
import argparse
import matplotlib.pyplot as plt
import argparse
import csv
import pandas as pd
import numpy as np
import re
import nltk
import tqdm
from tensorflow.keras import layers
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

def preprocess_text(sen):
        sentence = remove_tags(sen)
        #sentence = re.sub(r'[@%\\*=()/~#&\+á?\xc3\xa1\-\|\.\:\;\!\-\,\_\~\$\'\"]', '',sentence) #remove punctuation
        sentence = re.sub(r'\d+',' ', sentence)# remove number
        sentence = sentence.lower() #lower case
        sentence = re.sub(r'\s+', ' ', sentence) #remove extra space
        sentence = re.sub(r'\s+', ' ', sentence) #remove spaces
        sentence = re.sub(r"^\s+", '', sentence) #remove space from start
        sentence = re.sub(r'\s+$', '', sentence) #remove space from the end

        return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def load_embeddings(filename):
    embeddings_dictionary = dict()
    embedding_file = open(filename, encoding="utf8")
    
    for line in embedding_file:
        values = line.split()
        word = values[0]
        vector_dimensions = np.asarray(values[1:])
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

np.random.seed(42)

print('Loading data...')
reviews = pd.read_csv("data/USairline/Tweet.csv", encoding = 'utf8')

X = []
sentences = list(reviews['text'])
for sen in sentences:
    X.append(preprocess_text(sen))
    
#we need to convert our labels into digits    
y = reviews['airline_sentiment']
y = np.array(list(map(lambda x: 1 if x=="positive" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify=y, test_size=0.20, random_state=42)

#Preparing the Embedding Layer
#use the Tokenizer class from the keras.preprocessing.
#text module to create a word-to-index dictionary.
#In the word-to-index dictionary, each word in the corpus is used as a key, 
#while a corresponding unique index is used as the value for the key
tokenizer = preprocessing.text.Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# Adding 1 because of reserved 0 index
vocab_size = len(tokenizer.word_index) + 1
print('Vocab size: ', vocab_size)
maxlen = 100

X_train = preprocessing.sequence.pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = preprocessing.sequence.pad_sequences(X_test, padding='post', maxlen=maxlen)

embedding_file = args.emb_filename

embeddings_dictionary = load_embeddings(embedding_file)

embedding_matrix = np.zeros((vocab_size, 300))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

        
# Build and train model
precision_list = []        
recall_list = []
f1_list = []
roc_list = []
loss_list = []
acc_list = []

for j in range(10):
    print('Build model...')
    
    model = Sequential()
    model.add(layers.Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=maxlen , trainable=False))
    model.add(layers.LSTM(128,dropout=0.2, recurrent_dropout=0.2))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()

    print('This is %d round.' % (j))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.10, shuffle=False)

    yhat_probs = model.predict(X_test, verbose=1)
    # predict crisp classes for test set
    yhat_classes = model.predict_classes(X_test, verbose=1)
    # reduce to 1d array
    yhat_probs = yhat_probs[:, 0]
    yhat_classes = yhat_classes[:, 0]

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