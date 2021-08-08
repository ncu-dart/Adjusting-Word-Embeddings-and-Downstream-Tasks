import torch
from torch.utils.data import Dataset
import random
from tqdm import tqdm

class BERTDataset(Dataset):
    def __init__(self, corpus_path, vocab, seq_len, encoding="utf-8", corpus_lines=None, on_memory=True):
        self.vocab = vocab
        self.seq_len = seq_len
        self.on_memory = on_memory
        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.encoding = encoding

        with open(corpus_path, "r", encoding=encoding) as f:
            if self.corpus_lines is None and not on_memory:
                for _ in tqdm(f, desc="Loading Dataset", total=corpus_lines):
                    self.corpus_lines += 1

            if on_memory:
                self.lines = [line[:-1].split("\t")
                              for line in tqdm(f, desc="Loading Dataset", total=corpus_lines)]
                self.corpus_lines = len(self.lines)

        if not on_memory:
            self.file = open(corpus_path, "r", encoding=encoding)
            self.random_file = open(corpus_path, "r", encoding=encoding)

            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()

    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        t1, t2 = self.random_sent(item)
        t1 = self.random_word(t1)
        t2 = self.random_word(t2)
        
        bert_input = (t1 + t2)[:self.seq_len]
        syn_label = ([1 for _ in range(len(t1))] + [-1 for _ in range(len(t2))])[:self.seq_len]
        
        padding_input = [0 for _ in range(self.seq_len - len(bert_input))]
        padding_label = [0 for _ in range(self.seq_len - len(bert_input))]

        bert_input.extend(padding_input)
        syn_label.extend(padding_label)

        output = {"bert_input": bert_input,
                  "syn_label": syn_label,}

        return {key: torch.tensor(value) for key, value in output.items()}

    def random_word(self, sentence):
        tokens = sentence.split()

        for i, token in enumerate(tokens):
            tokens[i] = self.vocab.stoi.get(token, self.vocab.unk_index)

        return tokens

    # delete random_sent
    def random_sent(self, index):
        t1, t2 = self.get_corpus_line(index)
        return t1, t2

    def get_corpus_line(self, item):
        if self.on_memory:
            return self.lines[item][0], self.lines[item][1]
        else:
            line = self.file.__next__()
            if line is None:
                self.file.close()
                self.file = open(self.corpus_path, "r", encoding=self.encoding)
                line = self.file.__next__()

            t1, t2 = line[:-1].split("\t")
            return t1, t2

    # delete get_random_line
    def get_random_line(self):
        if self.on_memory:
            return self.lines[random.randrange(len(self.lines))][1]

        line = self.file.__next__()
        if line is None:
            self.file.close()
            self.file = open(self.corpus_path, "r", encoding=self.encoding)
            for _ in range(random.randint(self.corpus_lines if self.corpus_lines < 1000 else 1000)):
                self.random_file.__next__()
            line = self.random_file.__next__()
        return line[:-1].split("\t")[1]
