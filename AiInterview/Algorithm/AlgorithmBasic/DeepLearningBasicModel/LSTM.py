import torch
from torch import nn
from datasets import load_dataset
from collections import Counter


class LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        return self.fc(hidden)

class LSTM_Model(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout):
        super().__init__()
        self.lstm = LSTM(vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, bidirectional, dropout)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)

    def forward(self, text):
        embedded = self.embedding(text)
        output = self.lstm(embedded)
        return self.fc(output)

    def predict(self, text):
        return self.forward(text)  
    
    def train_model(self, train_data, test_data, epochs, learning_rate, batch_size):
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            for batch in train_dataloader:
                text, label = batch
                optimizer.zero_grad()
                predictions = self.forward(text)
                loss = criterion(predictions, label)
                loss.backward()
                optimizer.step()
            with torch.no_grad():
                test_loss = 0
                test_acc = 0
                for batch in test_dataloader:
                    text, label = batch
                    predictions = self.forward(text)
                    test_loss += criterion(predictions, label)
                    test_acc += (predictions.argmax(dim=1) == label).sum().item()
                test_loss /= len(test_dataloader)
                test_acc /= len(test_dataloader)
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        return self
    
    def evaluate(self, test_data, batch_size):
        test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
        criterion = nn.CrossEntropyLoss()
        test_loss = 0
        test_acc = 0
        for batch in test_dataloader:
            text, label = batch
            predictions = self.forward(text)
            test_loss += criterion(predictions, label)
            test_acc += (predictions.argmax(dim=1) == label).sum().item()
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)
        return test_loss, test_acc

if __name__ == "__main__":
    dataset = load_dataset("imdb")
    train_data = dataset["train"]
    test_data = dataset["test"]
    vocab = Counter()
    for text in train_data["text"]:
        vocab.update(text.split())
    vocab_size = len(vocab)
    embedding_dim = 100

