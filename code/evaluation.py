import argparse
import torch
from torch.utils.data import DataLoader
import os
import bcolz
import pickle
import operator
import scipy.stats
import numpy as np
from tqdm import tqdm
from model import BERT
from trainer import BERTTrainer
from dataset import BERTDataset, WordVocab

def LoadInputVector(wordvec, data, lookup):
    input1, input2 = [], []
    OOV = []
    for i in range(len(data[:,0])):
        try:
            input1.append(wordvec[data[i,0]])
            try:
                input2.append(wordvec[data[i,1]])
            except:
                input1.pop()
                print(wordvec[data[i,0]],wordvec[data[i,1]])
        except:
            print(wordvec[data[i,0]],wordvec[data[i,1]])

    return np.array(input1), np.array(input2)

def Evaluating_MEN_Spearman(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.spearmanr(output, np.array(data[:,2], dtype=float))[0], 4)

def Evaluating_MEN_Pearson(wordvec, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    output = []
    epsilon = 1e-5
    for i in range(len(input1)):
        output.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
    output = (np.array(output)).reshape(-1)
    return round(scipy.stats.pearsonr(output, np.array(data[:,2], dtype=float))[0], 4)

def Evaluating_MEN_Cosine(wordvec, new_wordVecs, data, lookup, dis='Euclidean'):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    input3, input4 = LoadInputVector(new_wordVecs, data, lookup)
    output_before = []
    output_after = []
    label = np.ones((len(input1),), dtype=np.int)
    target = np.ones((len(input1),), dtype=np.int)
    """
    elements in loss_dict
    loss_before: loss of word wmbedding before adjusted
    loss_after: loss of word wmbedding after adjusted
    syn: amounts of word pairs that are synonym
    move_syn: amount of consine similarity of synonyms that move toward 1
    move_syn_mean: mean of consine similarity move of synonyms
    ant: amounts of word pairs that are antonym
    move_ant: amount of consine similarity of antonyms that move toward -1
    move_ant_mean: mean of consine similarity move of antonyms
    """
    loss_dict = {'loss_before': 0, 'loss_after': 0, 'syn': 0, 'move_syn': 0, 'move_syn_mean': 0, 'ant': 0, 'move_ant': 0, 'move_ant_mean': 0} 
    
    for i in tqdm(range(len(input1))):
        if dis == 'Cosine':
            output_before.append(np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i])))
            output_after.append(np.dot(input3[i], input4[i])/(np.linalg.norm(input3[i])*np.linalg.norm(input4[i])))
        elif dis == 'Euclidean':
            output_before.append(np.sqrt(sum(pow(input1[i] - input2[i], 2))))
            output_after.append(np.sqrt(sum(pow(input3[i] - input4[i], 2))))
        
        if data[i][2] == 'syn':
            loss_dict['syn'] += 1
            loss_dict['move_syn_mean'] += output_after[i] - output_before[i]
            if dis == 'Cosine':
                if output_after[i] - output_before[i] > 0:
                    loss_dict['move_syn'] += 1
            elif dis == 'Euclidean':
                if output_after[i] - output_before[i] < 0:
                    loss_dict['move_syn'] += 1
        if data[i][2] == 'ant':
            loss_dict['ant'] += 1
            loss_dict['move_ant_mean'] += output_after[i] - output_before[i]
            target[i] = 0
            label[i] = -1
            if dis == 'Cosine':
                if output_after[i] - output_before[i] < 0:
                    loss_dict['move_ant'] += 1
            elif dis == 'Euclidean':
                if output_after[i] - output_before[i] > 0:
                    loss_dict['move_ant'] += 1
    output_before = torch.from_numpy((np.array(output_before)).reshape(-1))
    output_after = torch.from_numpy((np.array(output_after)).reshape(-1))
    label = torch.from_numpy(label)
    target = torch.from_numpy(target)
    
    if dis == 'Cosine':
        loss_dict['loss_before'] = ((torch.sub(target,torch.mul(output_before, label))).sum()/torch.abs(label).sum()).mean()
        loss_dict['loss_after'] = ((torch.sub(target,torch.mul(output_after, label))).sum()/torch.abs(label).sum()).mean()
    elif dis == 'Euclidean':
        loss_dict['loss_before'] = (torch.mul(output_before, label).sum()/torch.abs(label).sum()).mean()
        loss_dict['loss_after'] = (torch.mul(output_after, label).sum()/torch.abs(label).sum()).mean()
                                
    print("<{} Evaluation of test data>".format(dis))
    print("loss before: ", loss_dict['loss_before'])
    print("loss after: ", loss_dict['loss_after'])
    print("amount of syn: ", loss_dict['syn'])
    print("syn move toward 1: ", loss_dict['move_syn'])
    print("mean of move of syn: ", loss_dict['move_syn_mean'] / loss_dict['syn'])
    print("amount of ant: ", loss_dict['ant'])
    print("ant move toward -1: ", loss_dict['move_ant'])
    if loss_dict['ant'] == 0:
        print("mean of move of ant: ", 0)
    else:
        print("mean of move of ant: ", loss_dict['move_ant_mean'] / loss_dict['ant'])
    
"""
def Evaluating_MEN_Cosine(wordvec, new_wordVecs, data, lookup):
    input1, input2 = LoadInputVector(wordvec, data, lookup)
    input3, input4 = LoadInputVector(new_wordVecs, data, lookup)
    output = []
    count_syn = np.zeros((2,), dtype=np.int)
    move_syn = 0
    count_ant = np.zeros((2,), dtype=np.int)
    move_ant = 0
    for i in range(len(input1)):
        if float(data[i][2]) <= 0:
            count_ant[1] += 1
            move = ((np.dot(input3[i], input4[i])/(np.linalg.norm(input3[i])*np.linalg.norm(input4[i]))) - (np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i]))))
            move_ant += move
            if move <= 0:
                count_ant[0] += 1
        else:
            count_syn[1] += 1
            move = ((np.dot(input3[i], input4[i])/(np.linalg.norm(input3[i])*np.linalg.norm(input4[i]))) - (np.dot(input1[i], input2[i])/(np.linalg.norm(input1[i])*np.linalg.norm(input2[i]))))
            move_syn += move
            if move >= 0:
                count_syn[0] += 1
    return move_syn/count_syn[1], count_syn[0], count_syn[1], move_ant/count_ant[1], count_ant[0], count_ant[1]
"""

def print_word_vecs(wordVectors, outFileName):
    print("Writing down the vectors in", outFileName)
    outFile = open(outFileName, 'w', encoding='utf-8')  
    for word, values in wordVectors.items():
        outFile.write(word+' ')
        for val in wordVectors[word]:
            outFile.write('%.4f' %(val)+' ')
        outFile.write('\n')      
    outFile.close()

parser = argparse.ArgumentParser()

parser.add_argument("-ep", "--emb_path", required=True, type=str, help="filepath of the original pre-trained embeddings")
parser.add_argument("-vp", "--vocab_path", required=True, type=str, help="filepath of vocabulary")
parser.add_argument("-tp", "--test_path", required=True, type=str, help="filepath of test data for evaluation")
parser.add_argument("-op", "--output_path", required=True, type=str, help="filepath for saving embeddings")

args = parser.parse_args()

# Read Vocab
print("Loading Vocab", args.vocab_path)
vocab = WordVocab.load_vocab(args.vocab_path)
print("Vocab Size: ", len(vocab))

word2idx = vocab.stoi
idx2word = vocab.itos


# Read WordVec
NormRead = True
wordVecs = {}
with open(args.emb_path, 'r', encoding='utf-8') as fileObject:
    for line in fileObject:
        tokens = line.strip().lower().split()
        try:
            wordVecs[tokens[0]] = np.fromiter(map(float, tokens[1:]), dtype=np.float64)
            if NormRead:
                wordVecs[tokens[0]] = wordVecs[tokens[0]] / np.sqrt((wordVecs[tokens[0]]**2).sum() + 1e-5)
        except:
            pass


# Read Model Raw Output
new_wordVecs = {}
for key,val in wordVecs.items():
    new_wordVecs[key] = val

import pickle
for i in tqdm(range(int(len([name for name in os.listdir('../output/embeddings/raw/')])/2))):
    inp = pickle.load(open('../output/embeddings/raw/result_input_iter{}.pkl'.format(i),'rb')).cpu().numpy()
    out = pickle.load(open('../output/embeddings/raw/result_output_iter{}.pkl'.format(i),'rb')).cpu().detach().numpy()
    for j in range(inp.shape[0]):
        for x,y in zip(inp[j],out[j]):
            try:
                new_wordVecs[idx2word[x]] = np.vstack((new_wordVecs[idx2word[x]],y))
            except:
                pass

for key,val in new_wordVecs.items():
    if len(val.shape)>1:
        new_wordVecs[key] = (sum(val)-wordVecs[key])/val.shape[0]

if NormRead:
    for key,val in new_wordVecs.items():
        new_wordVecs[key] = val / np.sqrt((val**2).sum() + 1e-5)


#Word Sim Task
#'MEN_3k', 'SL_999', 'WS_353', 'RG_65', 'MTURK-771', 'SV_3500', 'WS_similarity', 
correlation_tasks = ['MEN_3k', 'SL_999', 'WS_353', 'RG_65', 'SV_3500_only_syn', 'SV_3500', 'SV_3500_ant', 'SV_3500_3labels', 'WS_353_only_syn', 'WS_353', 'WS_353_ant', 'WS_353_3labels']
for i in correlation_tasks:
    with open('../data/testsets/{}.txt'.format(i), 'r', encoding='utf-8') as fp_men:
        fp_men_ = fp_men.readlines()
        data_men = [row.strip().split(' ') for row in fp_men_]
        data_men = np.array(data_men)
    
    word_to_idx_men = {}
    idx = 0

    for w in data_men[:,0]:
        try: word_to_idx_men[w]
        except KeyError:
            word_to_idx_men[w] = idx
            idx = idx+1
     
    for w in data_men[:,1]:
        try: word_to_idx_men[w]
        except KeyError:
            word_to_idx_men[w] = idx
            idx = idx+1
    
    word_to_idx_men = sorted(word_to_idx_men.items(), key=operator.itemgetter(1))
    lookup_men = dict(word_to_idx_men)
    
    # print('<{} Dataset>'.format(i))
    # print("Spearman Before :", Evaluating_MEN_Spearman(wordVecs, data_men, lookup_men))
    # print("         After  :", 
    print(Evaluating_MEN_Spearman(new_wordVecs, data_men, lookup_men))
    # print("Pearson  Before :", Evaluating_MEN_Pearson(wordVecs, data_men, lookup_men))
    # print("         After  :", 
    print(Evaluating_MEN_Pearson(new_wordVecs, data_men, lookup_men))

with open(args.test_path, 'r', encoding='utf-8') as fp_men:
    fp_men_ = fp_men.readlines()
    data_men = [row.strip().split(' ') for row in fp_men_]
    data_men = np.array(data_men)

word_to_idx_men = {}
idx = 0

for w in data_men[:,0]:
    try: word_to_idx_men[w]
    except KeyError:
        word_to_idx_men[w] = idx
        idx = idx+1

for w in data_men[:,1]:
    try: word_to_idx_men[w]
    except KeyError:
        word_to_idx_men[w] = idx
        idx = idx+1

word_to_idx_men = sorted(word_to_idx_men.items(), key=operator.itemgetter(1))
lookup_men = dict(word_to_idx_men)

Evaluating_MEN_Cosine(wordVecs, new_wordVecs, data_men, lookup_men, dis='Cosine')
Evaluating_MEN_Cosine(wordVecs, new_wordVecs, data_men, lookup_men, dis='Euclidean')

print_word_vecs(new_wordVecs, args.output_path)
