# Adjusting Word Embeddings and Downstream Tasks

# Introduction
We adjust the pre-trained word embeddings through a self-attention mechanism so that the word embeddings can preserve the relationship between synonyms and antonyms in the lexicon, and evaluate the adjusted word embeddings the through NLP downstream tasks.

Paper Links: []()

# Run

## Adjusting Word Embeddings

### main<span></span>.py
* Train self-attention model and adjust pre-trained word embeddings.
* Refer to the chapter 3.1.3 in the paper for the input of loss1/loss2/alpha

Usage:
```
$python main.py -ep <filepath of pre-trained embeddings> \
                -en <filename of pre-trained embeddings> \
                -lp <filepath of lexicons> \
                -vp <filepath of vocabulary> \
                -op <filepath to save model> \
                -loss1 <method to use for the first target of the loss function> ('Cosine' or 'Euclidean') \
                -loss2 <method to use for the second target of the loss function> ('MSE' or 'Euclidean') \
                -alpha <hyperparameter for the loss function>
```
Example:
```
$python main.py -ep ../data/embeddings/GloVe/glove.6B.300d.txt \
                -en glove.6B.300d \
                -lp ../data/lexicons/wordnet_syn_ant.txt \
                -vp ../data/embeddings/GloVe/glove.6B.300d.txt.vocab.pkl \
                -op ../output/model/listwise.model \
                -loss1 Euclidean \
                -loss2 Euclidean \
                -alpha 0.3          
```

### evaluation<span></span>.py
* Compare the performance of un-adjusted and adjusted word embeddings on test data.  
* Transfer the raw output to GloVe format.

Usage:
```
$python evaluation.py -ep <filepath of the original pre-trained embeddings> \
                      -vp <filepath of vocabulary> \
                      -tp <filepath of test data for evaluation> \
                      -op <filepath for saving embeddings>
```
Example:
```
$python evaluation.py -ep ../data/embeddings/GloVe/glove.6B.300d.txt \
                      -vp ../data/embeddings/GloVe/glove.6B.300d.txt.vocab.pkl \
                      -tp ../data/lexicons/wordnet_syn_ant_test_pair.txt \
                      -op ../output/embeddings/Listwise_Vectors.txt
```

## NLP Downstream Tasks

### IMDb Task: LSTM_IMDb<span></span>.py
* Compare the performance of un-adjusted and adjusted word embeddings on IMDb.

Usage:
```
$python LSTM_IMDb.py -en <filename of embeddings> \
                     -op <filepath for saving the result>
```
Example:
```
$python LSTM_IMDb.py -en ../output/embeddings/Listwise_Vectors.txt \
                     -op ../output/IMDb/IMDb_result.txt
```

### US Airline Task: LSTM_USairline<span></span>.py
* Compare the performance of un-adjusted and adjusted word embeddings on US Airline.

Usage:
```
$python LSTM_USairline.py -en <filename of embeddings> \
                          -op <filepath for saving the result>
```
Example:
```
$python LSTM_USairline.py -en ../output/embeddings/Listwise_Vectors.txt \
                          -op ../output/USairline/USairline_result.txt
```

# Datasets

## Embeddings
* Pretrained word embeddings filtered by 50K frequent words in GloVe format.
* Files end with '_train.txt' are for training, while files end with '_test_pair.txt' are for testing.

Data format:
```
word1 -0.09611 -0.25788 ... -0.092774  0.39058
word2 -0.24837 -0.45461 ...  0.15458  -0.38053
```

## Lexicons
* Synonyms and antonyms retrieved from dictionary.

Data format:
```
word1 syn1 ... synn \t ant1 ... antn 
word2 syn1 ... synn \t ant1 ... antn 
```

# Reference
1. https://github.com/mfaruqui/retrofitting
2. https://github.com/codertimo/BERT-pytorch
3. https://nlp.seas.harvard.edu/2018/04/03/attention.html
