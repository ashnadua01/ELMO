import pandas as pd
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset

# nltk.download('punkt')

class CustomDataset(Dataset):
    def __init__(self, data, given_vocab=None):
        self.data = data
        self.tokens = [self.word_tokenizer(doc) for doc in self.data['Description']]
        self.vocab = self.build_vocab(given_vocab)
        self.padding()

    def build_vocab(self, given_vocab=None):
        if given_vocab == None:
            vocab = set()
            for tokens in tqdm(self.tokens, total=len(self.tokens)):
                for token in tokens:
                    vocab.add(token)

            vocab = {word: i for i, word in enumerate(vocab, start=1)}
            vocab['<unk>'] = 0

            return vocab
        else:
            return given_vocab

    def word_tokenizer(self, sentence):
        tokens = word_tokenize(sentence)
        tokens = [token.lower() for token in tokens]
        return ['<start>'] + tokens + ['<end>']
    
    def convert_to_tensor(self, sentence):
        return torch.tensor([self.vocab[word] if word in self.vocab else self.vocab['<unk>'] for word in sentence])

    def padding(self):
        self.tokens = [self.convert_to_tensor(doc) for doc in self.tokens]
        self.padded_sequences = pad_sequence(self.tokens, batch_first=True, padding_value=len(self.vocab))
        self.classes = np.unique(self.data['Class Index'])

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        label = self.data.loc[idx, 'Class Index']
        tokens = self.padded_sequences[idx]
        return tokens, torch.tensor(label, dtype=torch.int64)
