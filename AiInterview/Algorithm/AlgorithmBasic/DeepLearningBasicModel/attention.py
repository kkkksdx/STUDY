from datasets import load_dataset
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

full_dataset = load_dataset("wmt/wmt17","zh-en", split="train")

def is_high_quality(x):
    en = x["translation"]["en"]
    zh = x["translation"]["zh"]
    if not en or not zh:
        return False
    pass

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.W_Q = nn.Linear(hidden_size, hidden_size,bias=False)
        self.W_K = nn.Linear(hidden_size, hidden_size,bias=False)
        self.W_V = nn.Linear(hidden_size, hidden_size,bias=False)


    def forward(self, encoder_outputs, decoder_outputs):
        Q = self.W_Q(decoder_outputs.unsqueeze(1))#(batch_size, 1, hidden_size)
        K = self.W_K(encoder_outputs)
        V = self.W_V(encoder_outputs)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.hidden_size)
        attn_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attn_weights, V)
        return context.squeeze(1), attn_weights

class Encoder(nn.Module):
    def __init__(self, hidden_size, n_layers, n_heads):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, n_heads) for _ in range(n_layers)])

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x

class Decoder(nn.Module):
    def __init__(self, hidden_size, n_layers, n_heads):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(max_length, hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(hidden_size, n_heads) for _ in range(n_layers)])
        
    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_embedding(x)
        for layer in self.layers:
            x = layer(x)
        return x







    