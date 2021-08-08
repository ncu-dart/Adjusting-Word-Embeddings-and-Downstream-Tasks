import argparse
import torch
from torch.utils.data import DataLoader
import bcolz
import pickle
import numpy as np
from model import BERT
from trainer import BERTTrainer
from dataset import BERTDataset, WordVocab

def ReadWordVec(filepath, filename, emb_dim):
    try:
        vectors = bcolz.open('./cache/'+filename+'.dat')[:]
        words = pickle.load(open('./cache/'+filename+'_words.pkl', 'rb'))
        word2idx = pickle.load(open('./cache/'+filename+'_idx.pkl', 'rb'))

        glove = {w: vectors[word2idx[w]] for w in words}

    except:
        words = []
        idx = 0
        word2idx = {}
        vectors = bcolz.carray(np.zeros(1), rootdir='./cache/'+filename+'.dat', mode='w')

        with open(filepath, 'rb') as f:
            for l in f:
                line = l.decode().split()
                word = line[0]
                words.append(word)
                word2idx[word] = idx
                idx += 1
                try:
                    vect = np.array(line[1:]).astype(np.float)
                except:
                    print(line)
                vectors.append(vect)

        vectors = bcolz.carray(vectors[1:].reshape((idx, emb_dim)), rootdir='./cache/'+filename+'.dat', mode='w')
        vectors.flush()
        pickle.dump(words, open('./cache/'+filename+'_words.pkl', 'wb'))
        pickle.dump(word2idx, open('./cache/'+filename+'_idx.pkl', 'wb'))
        
        vectors = bcolz.open('./cache/'+filename+'.dat')[:]
        words = pickle.load(open('./cache/'+filename+'_words.pkl', 'rb'))
        word2idx = pickle.load(open('./cache/'+filename+'_idx.pkl', 'rb'))

        glove = {w: vectors[word2idx[w]] for w in words}

    target_vocab = vocab.itos
    matrix_len = len(target_vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))

    for i, word in enumerate(target_vocab):
        try:
            weights_matrix[i] = glove[word]
        except KeyError:
            weights_matrix[i] =  np.zeros(emb_dim)

    return torch.Tensor(weights_matrix)


parser = argparse.ArgumentParser()

parser.add_argument("-ep", "--emb_path", required=True, type=str, help="filepath of pre-trained embeddings")
parser.add_argument("-en", "--emb_filename", required=True, type=str, help="filename of pre-trained embeddings")
parser.add_argument("-lp", "--lexicon_path", required=True, type=str, help="filepath of lexicons")
parser.add_argument("-vp", "--vocab_path", required=True, type=str, help="filepath of vocabulary")
parser.add_argument("-op", "--output_path", required=True, type=str, help="filepath for saving model")

parser.add_argument("-loss1", "--loss1", type=str, default='Euclidean', help="method to use for the first target of the loss function ")
parser.add_argument("-loss2", "--loss2", type=str, default='MSE', help="method to use for the second target of the loss function ")
parser.add_argument("-alpha", "--alpha", type=float, default='0.5', help="hyperparameter for the loss function ")

parser.add_argument("-d", "--emb_dim", type=int, default=300, help="size of pre-trained embeddings")
parser.add_argument("-l", "--layers", type=int, default=1, help="number of layers")
parser.add_argument("-a", "--attn_heads", type=int, default=1, help="number of attention heads")
parser.add_argument("-s", "--seq_len", type=int, default=10, help="maximum input sequence len")

parser.add_argument("-b", "--batch_size", type=int, default=64, help="number of batch_size")
parser.add_argument("-e", "--epochs", type=int, default=20, help="number of epochs")
parser.add_argument("-w", "--num_workers", type=int, default=4, help="dataloader worker size")

parser.add_argument("--with_cuda", type=bool, default=True, help="training with CUDA: true, or false")
parser.add_argument("--log_freq", type=int, default=100, help="printing loss every n iter: setting n")
parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")
parser.add_argument("--on_memory", type=bool, default=True, help="Loading on memory: true or false")

parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

args = parser.parse_args()

print("Loading Vocab", args.vocab_path)
vocab = WordVocab.load_vocab(args.vocab_path)
print("Vocab Size: ", len(vocab))

print("Loading Train Dataset", args.lexicon_path)
train_dataset = BERTDataset(args.lexicon_path, vocab, seq_len=args.seq_len,
                            corpus_lines=args.corpus_lines, on_memory=args.on_memory)

print("Creating Dataloader")
train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

print("Reading Word Vectors")
weights_matrix = ReadWordVec(args.emb_path, args.emb_filename, args.emb_dim)

print("Building Model")
bert = BERT(len(vocab), weights_matrix, hidden=args.emb_dim, n_layers=args.layers, attn_heads=args.attn_heads)

print("Creating Trainer")
trainer = BERTTrainer(bert, len(vocab), args.seq_len, train_dataloader=train_data_loader,
                        lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                        loss1=args.loss1, loss2=args.loss2, alpha=args.alpha,
                        with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

print("loss1: ", args.loss1)
print("loss2: ", args.loss2)
print("alpha: ", args.alpha)

print("Training Start")
for epoch in range(args.epochs):
    trainer.train(epoch)
    trainer.save(epoch, args.output_path)

trainer.test(1)