import pandas as pd
import torch
from tqdm import tqdm
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader

from data import CustomDataset
import argparse
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ELMo(nn.Module):
    def __init__(self, vocab_size, e_dim, h_dim):
        super(ELMo, self).__init__()
        self.vocab_size = vocab_size
        self.e_dim = e_dim
        self.h_dim = h_dim
        self.pad_token = torch.tensor(self.vocab_size)
        self.embedding = nn.Embedding(self.vocab_size + 1, self.e_dim, padding_idx=self.vocab_size)

        # forward and backward LSTMs for the first layer
        self.lstm_for1 = nn.LSTM(self.e_dim, self.h_dim, batch_first=True)
        self.lstm_for2 = nn.LSTM(self.h_dim, self.h_dim, batch_first=True)

        # forward and backward LSTMs for the second layer
        self.lstm_back1 = nn.LSTM(self.e_dim, self.h_dim, batch_first=True)
        self.lstm_back2 = nn.LSTM(self.h_dim, self.h_dim, batch_first=True)

        # embedding layer for combining layer outputs
        self.weights = nn.Embedding(1, 3)
        self.weights.weight = nn.Parameter(torch.ones(1, 3) / 3, requires_grad=True)

        # padding
        self.end_padding = torch.zeros(self.h_dim, requires_grad=False).unsqueeze(0).unsqueeze(0).to(device)
        self.linear = nn.Linear(2 * self.h_dim, self.vocab_size)
    
    def for_output(self, x, x_rev):
        # Embed the input sequences
        forward = self.embedding(x)
        reverse = self.embedding(x_rev)

        # first layer LSTMs
        lstm1_out_forward, _ = self.lstm_for1(forward)
        lstm2_out_forward, _ = self.lstm_for2(lstm1_out_forward)
        
        # second layer LSTMs
        lstm1_out_backward, _ = self.lstm_back1(reverse)
        lstm1_out_backward = lstm1_out_backward.flip(1)
        lstm2_out_backward, _ = self.lstm_back2(lstm1_out_backward)
        lstm2_out_backward = lstm2_out_backward.flip(1)

        return lstm1_out_forward, lstm1_out_backward, lstm2_out_forward, lstm2_out_backward, forward

    def pad(self, fwd, bwd, output_padding, padding_indices):
        fwd = torch.cat((output_padding, fwd), dim=1)
        bwd = torch.cat((bwd, output_padding, output_padding), dim=1)
        bwd[torch.arange(len(fwd)).to(device), padding_indices, :] = torch.zeros(len(fwd), bwd.shape[2]).to(device)
        return torch.cat((fwd, bwd[:, 1:]), dim=2)

    def forward(self, x):
        x_rev = x.flip(1)
        padding_indices = torch.sum(x != self.pad_token, dim=1).to(device)

        lstm1_fwd, lstm1_bwd, lstm2_fwd, lstm2_bwd, forward = self.for_output(x, x_rev)

        # Prepare padding for layer outputs
        output_padding = self.end_padding.expand(lstm2_fwd.shape[0], 1, lstm2_fwd.shape[2])

        # Padding and concatenation for second layer outputs
        lstm2_out = self.pad(lstm2_fwd, lstm2_bwd, output_padding, padding_indices)
        lstm1_out = self.pad(lstm1_fwd, lstm1_bwd, output_padding, padding_indices)

        # Weighted sum of layer outputs and embeddings
        norm = self.weights(torch.zeros(1).long().to(device)) 
        weighted_output = norm[0, 0] * lstm1_out + norm[0, 1] * lstm2_out + norm[0, 2] * torch.cat((torch.cat((output_padding, output_padding), dim=2), forward), dim=1)

        output_probs = self.linear(weighted_output)
        return output_probs[:, :-1, :]
    
    def pretrain(self, train_loader, learning_rate=0.01, num_epochs=10):
        print("\n-----------Pretraining Model-----------\n")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index = self.vocab_size)
        
        for epoch in range(num_epochs):
            self.train()
            train_loss = 0
            for i, (input, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
                optimizer.zero_grad()
                input = input.to(device)
                outputs = self(input)
                loss = criterion(outputs.reshape(-1, outputs.size(2)), input.view(-1))
                train_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f"Epoch {epoch}, Train loss: {train_loss / len(train_loader)}")
        
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
        
    def generate_embeddings(self, input):
        x_rev = input.flip(1).to(input.device)  # Ensure x_rev is on the same device as input
        lstm1_out_forward, lstm1_out_backward, lstm2_out_forward, lstm2_out_backward, forward = self.for_output(input, x_rev)
        lstm1_out = torch.cat((lstm1_out_forward, lstm1_out_backward), dim=2)
        lstm2_out = torch.cat((lstm2_out_forward, lstm2_out_backward), dim=2)

        return lstm1_out, lstm2_out, forward

def evaluate(model, test_loader):
    print("\n-----------Evaluating Model-----------\n")
    model.eval()
    with torch.no_grad():
        acc = 0
        for input, target in tqdm(test_loader, desc = "Testing"):
            input = input.to(device)
            outputs = model(input.to(device))
            predicted = torch.argmax(outputs, dim = 2).view(-1)
            actual = input.view(-1)
            pad_indices = actual == vocab_size
            count_pads = torch.sum(pad_indices)
            predicted[pad_indices] = vocab_size
            acc += (torch.sum(predicted == actual) - count_pads) / (len(actual) - count_pads)

        print(f"Test Accuracy: {acc / len(test_loader)}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain', type=bool, default=False, help='Use pretrained model')
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

    if args.pretrain:
        elmo_model = ELMo(vocab_size, e_dim, h_dim).to(device)
        elmo_model.load_state_dict(torch.load('./model_checkpoints/bilstm.pt', map_location=device))
        evaluate(elmo_model, test_loader)
    else:
        elmo_model = ELMo(vocab_size, e_dim, h_dim).to(device)
        elmo_model.pretrain(train_loader, num_epochs=10)
        evaluate(elmo_model, test_loader)
