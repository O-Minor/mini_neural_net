# using this tutorial
# https://pub.towardsai.net/build-the-smallest-llm-from-scratch-with-pytorch-and-generate-pok%C3%A9mon-names-fcff7dcc7e36
# goal of tutorial: generate pokemon names
# goal of my demo: generate ai bot names

# imports
import pandas as pd
import torch
import string
import numpy as np
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt

data = pd.read_csv('pokemon.csv')["Name"]
words = data.to_list()
for i in range(len(words)):
    words[i] = words[i].lower()
# print(words[:8])
# ['Bulbasaur', 'Ivysaur', 'Venusaur', 'VenusaurMega Venusaur', 'Charmander', 'Charmeleon', 'Charizard', 'CharizardMega Charizard X']

# BUILD VOCABULARY
chars = sorted(list(set(' '.join(words))))
#stoi = string to int (maps chars to unique ints)
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0 # dot represents end of word
# itos = int to string
itos = {i:s for s,i in stoi.items()}
# print(stoi)
# {' ': 1, '%': 2, "'": 3, '-': 4, '.': 0, '0': 6, '2': 7, '5': 8, 'A': 9, 'B': 10, 'C': 
# had to remove: .s on Mr. Mime and Mime Jr., 
# gender symbols on Nidorans, 2 in Proygon2, on Zygarde removed "50% Forme"
# dashes in Ho-oh and Porygon-Z replaced with spaces
# print(itos)
# {1: ' ', 2: '%', 3: "'", 4: '-', 0: '.', 6: '0', 7: '2', 8: '5', 9: 'A', 10: 'B', 11: '

# MAKE N GRAMS
block_size = 3 # context length (good for japanese because hiriganas are 2-3 latin chars)
def build_dataset(words):
    X, Y = [], []
    for w in words:
        context = [0] * block_size # star with a blank context
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # shift and append new character
    return torch.tensor(X), torch.tensor(Y)
X, Y = build_dataset(words[:int(0.8*len(words))]) # use 80% for training data
# oliver notes: unsure if this randomizes order but it would be good if it did 
# otherwise you'd like train on nothing with Z names or something
print(X.shape, Y.shape) # check shapes of training data
# torch.Size([6155, 3]) torch.Size([6155])

# INITIALIZE PARAMETERS with random values
# tutorial uses 27 as number of characters, I am using 28 because é is included
g  = torch.Generator()
C  = torch.randn((28, 10), generator=g) # embedding layer
# W: weight, b: bias
W1 = torch.randn((30, 200), generator=g) # first linear layer
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, 28), generator=g) # second linear layer
b2 = torch.randn(28, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True # enables backpropogation, to adjust values

for i in range(100000):
    # get a mini-batch of 32 random samples from training data
    ix = torch.randint(0, X.shape[0], (32,))
    emb = C[X[ix]]
    # pass embedding through hidden layer W1 "with tanh activate"
    h = torch.tanh(emb.view(-1, 30) @ W1 + b1)
    # "calculate logits for output"
    # oliver note: logits is .-` shaped  
    # it looks like x^3 but starts down and shifted to 0.5
    # it is ln( x/(1-x) )
    # it sticks any number into range 0-1 for probabilities
    logits = h @ W2 + b2
    # using cross entropy as the loss method to get less bad each step
    loss = F.cross_entropy(logits, Y[ix])
    for p in parameters:
        p.grad = None
    loss.backward()
    for p in parameters:
        p.data -= 0.1 * p.grad

# Find the Probability of the Next Character
# <break for data cleaning>
