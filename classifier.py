import pandas as pd
import numpy as np
import torch
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import nltk
import random
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


from nltk.tokenize import word_tokenize
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader, Dataset
from collections import Counter

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

from data import CustomDataset
from ELMO import ELMo
import argparse
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMClassifier(nn.Module):
    def __init__(self, elmo_model, h_dim, e_dim, classes, combine='trainable'):
        super(LSTMClassifier, self).__init__()
        self.elmo_model = elmo_model
        self.h_dim = h_dim
        self.classes = classes
        self.h_dim = h_dim
        self.e_dim = e_dim
        self.combine = combine
        self.vocab_size = elmo_model.vocab_size
        
        for param in self.elmo_model.parameters():
            param.requires_grad = False
        
        self.lstm = nn.LSTM(input_size=self.e_dim, hidden_size=self.h_dim, batch_first=True)        
        self.output_layer = nn.Linear(self.h_dim, classes)

        if combine == 'trainable':
            self.embedding_weights = nn.Parameter(torch.ones(3) / 3)
        elif combine == 'frozen':
            self.embedding_weights = torch.randn(3, requires_grad=False)
        elif combine == 'function':
            self.layer = nn.Sequential(
                nn.Linear(3*self.e_dim, self.e_dim), nn.ReLU())
        else:
            print("Incorrect option")
            return
        
    def forward(self, X, classification='mean'):
        lstm1, lstm2, embedding_layer = self.elmo_model.generate_embeddings(X)

        embedding_types = ['trainable', 'frozen']
        if self.combine in embedding_types:
            input = sum(self.embedding_weights[i] * layer for i, layer in enumerate([lstm1, lstm2, embedding_layer]))
        else:
            input = self.layer(torch.cat((lstm1,lstm2,embedding_layer), dim = 2))
        
        output, _ = self.lstm(input)
        lengths = (X != self.vocab_size).sum(dim=1)
        
        hidden_states = [output[idx, lengths[idx] - 1].unsqueeze(0) if classification == 'last' else torch.mean(output[idx, 0:lengths[idx]], dim=0).unsqueeze(0) for idx, _ in enumerate(X)]
        hidden_states = torch.cat(hidden_states, dim=0)
        final = self.output_layer(hidden_states)
        
        return final
    
    def train_model(self, train_loader, classification="mean", num_epochs=10, learning_rate=0.01):
        print('/n----------Training Model----------/n')
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for input, target in tqdm(train_loader, total=len(train_loader)):
                optimizer.zero_grad()
                input = input.to(device)
                target = target.to(device)
                
                outputs = self(input,classification)

                # classes are 1, 2, 3, 4 (prediction will be 0, 1, 2, 3)
                loss = criterion(outputs, target-1)
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
                
            print(f"Epoch {epoch}, Train loss: {train_loss / len(train_loader)}")

    def evaluate_model(self, eval_loader, flag=0):
        print('/n----------Evaluating Model----------/n')
        self.eval()

        all_labels = []
        all_predictions = []

        with torch.no_grad():  # No need to track gradients
            for input_batch, output_batch in eval_loader:
                input_batch = input_batch.to(device)
                output_batch = output_batch.to(device)
                output_batch -= 1

                outputs = self(input_batch)
                _, predicted = torch.max(outputs, 1)

                all_labels.extend(output_batch.tolist())
                all_predictions.extend(predicted.tolist())

        accuracy = accuracy_score(all_labels, all_predictions)
        f1_micro = f1_score(all_labels, all_predictions, average='micro')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')

        print("Accuracy: ", accuracy)
        print("F1 Macro: ", f1_micro)
        print("F1 Micro: ", f1_macro)
        print("Precision: ", precision)
        print("Recall: ", recall)

        if flag == 1:
            cm = confusion_matrix(all_labels, all_predictions)
            class_names = [1, 2, 3, 4]

            plt.figure(figsize=(10,10))
            sns.heatmap(cm, annot=True, square=True, fmt='d', cmap='Blues', 
                        xticklabels=class_names, yticklabels=class_names)

            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
            
        return accuracy, f1_micro, f1_macro, precision, recall
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        map_location = 'cpu' if not torch.cuda.is_available() else None
        self.load_state_dict(torch.load(path, map_location=map_location))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trained', type=bool, default=False, help='Use trained classification model')
    parser.add_argument('--type', type=str, default='trainable', choices=['trainable', 'frozen', 'function'], help='Type: trainable, frozen, or function')

    args = parser.parse_args()

    with open('./model_checkpoints/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    with open('./model_checkpoints/test_data.pkl', 'rb') as f:
        test_data = pickle.load(f)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    vocab = train_data.vocab
    vocab_size = len(vocab)

    h_dim = 128
    e_dim = 256

    elmo_model = ELMo(vocab_size, e_dim, h_dim).to(device)
    elmo_model.load_state_dict(torch.load('./model_checkpoints/bilstm.pt', map_location=device))

    if args.trained:
        if args.type == 'trainable':
            model = LSTMClassifier(elmo_model, 256, e_dim, 4, 'trainable').to(device)
            model.load_model('./model_checkpoints/classifier_trainable.pt')
        elif args.type == 'frozen':
            model = LSTMClassifier(elmo_model, 256, e_dim, 4, 'frozen').to(device)
            model.load_model('./model_checkpoints/classifier_frozen.pt')
        if args.type == 'function':
            model = LSTMClassifier(elmo_model, 256, e_dim, 4, 'function').to(device)
            model.load_model('./model_checkpoints/classifier_function.pt')
    else:
        model = LSTMClassifier(elmo_model, 256, e_dim, 4, args.type).to(device)
        model.train_model(train_loader)

    model.evaluate_model(test_loader, flag=0)

