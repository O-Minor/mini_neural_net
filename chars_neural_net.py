# based on this tutorial
# https://pub.towardsai.net/build-the-smallest-llm-from-scratch-with-pytorch-and-generate-pok%C3%A9mon-names-fcff7dcc7e36
# goal of tutorial: generate pokemon names
# goal of my demo: generates words similar to the .txt(s) trained on

# imports
import pandas as pd
import torch
import string
import numpy as np
import re
import torch.nn.functional as F
import matplotlib.pyplot as plt

# STEP 1: PARSE DATA TO TRAIN ON
# the training set of words
words = []

# add_txt_to_words
# input: a .txt file or really any data stream that open().read() can parse, current words list
# output: words updated with each word (split on spaces) in the txt file as a word
def add_txt_to_words(in_txt, words, delim=' '):
    in_text = open(in_txt).read()
    # print(in_text)
    in_text=in_text.replace('\n',' ')
    # print(in_text) # for testing if /n repl works ✔️
    # split text into words on spaces
    temp_list = in_text.split(delim)
    words = words + temp_list
    return words

# add_col_to_words
# input: a csv file and the specified column name to import, current words list
# output: words updated with each row of the column as a word
def add_col_to_words(in_csv, col_name, words):
    data = pd.read_csv(in_csv)[col_name]
    temp_list = data.to_list()
    words = words + temp_list
    return words

# add_csv_to words
# input: a csv file with only one column, likely of text, current words list
# output: words updated with each entry of the csv as a word
def add_csv_to_words(in_csv, words):
    data = pd.read_csv(in_csv)[0]
    temp_list = data.to_list()
    words = words + temp_list
    return words

words = add_col_to_words('pokemon.csv', "Name", words)
print(words[-8:])
words = add_txt_to_words('ai_names_real_fic.csv', words, ',')
print(words[-8:])
words = add_txt_to_words('blorbo2.txt', words)
print(words[-8:])

# BUILD VOCABULARY
# by this point have a list of words called words
# research or ask Alex if it would even make sense to make another version that is word by word
# re: curse of dimentionality

chars = sorted(list(set(' '.join(words))))
#stoi = string to int (maps chars to unique ints)
stoi = {s:i+1 
        for i,s in enumerate(chars)}
stop_char = '␄'
stoi[stop_char] = 0 # character to represent end of what should be generated
chrs_len = len(stoi.items())
# itos = int to string
itos = {i:s 
        for s,i in stoi.items()}
# print("strs to ints\n",stoi)
# {' ': 1, '%': 2, "'": 3, '-': 4, '.': 0, '0': 6, '2': 7, '5': 8, 'A': 9, 'B': 10, 'C': 

# print(itos)
# {1: ' ', 2: '%', 3: "'", 4: '-', 0: '.', 6: '0', 7: '2', 8: '5', 9: 'A', 10: 'B', 11: '

# MAKE N GRAMS
block_size = 2 # context length (good for japanese because hiriganas are 2-3 latin chars)
ten_blocks = block_size * 10
def build_dataset(words, block_size):
    ten_blocks = block_size * 10
    X, Y = [], []
    for w in words:
        context = [0] * block_size # star with a blank context
        for ch in w + stop_char:
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # shift and append new character
    return torch.tensor(X), torch.tensor(Y)
X, Y = build_dataset(words[:int(0.8*len(words))]) # use 80% for training data
# oliver notes: unsure if this randomizes order but it would be good if it did 
# otherwise you'd like train on nothing with Z names or something
# print(X.shape, Y.shape) # check shapes of training data
# torch.Size([6155, 3]) torch.Size([6155])

# INITIALIZE PARAMETERS with random values
# generalized the tutorial's 27 to chrs_len which is length of stoi
g  = torch.Generator()
C  = torch.randn((chrs_len, 10), generator=g) # embedding layer
# W: weight, b: bias
W1 = torch.randn((ten_blocks, 200), generator=g) # first linear layer
b1 = torch.randn(200, generator=g)
W2 = torch.randn((200, chrs_len), generator=g) # second linear layer
b2 = torch.randn(chrs_len, generator=g)

parameters = [C, W1, b1, W2, b2]
for p in parameters:
    p.requires_grad = True # enables backpropogation, to adjust values

for i in range(100000):
    # get a mini-batch of 32 random samples from training data
    ix = torch.randint(0, X.shape[0], (32,))
    emb = C[X[ix]]
    # pass embedding through hidden layer W1 "with tanh activate"
    h = torch.tanh(emb.view(-1, ten_blocks) @ W1 + b1)
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

pik_input = "pik" # example input to get probabilities of next characters
input_chars = pik_input
#convert input characters to indices based on str to int (stoi) the char->index map
# ensure context fits block size
context = [stoi.get(char,0) 
           for char in input_chars][-block_size:]
# pad if shorter than block size
context = [0] * (block_size - len(context)) + context
# oliver note: need to do this for mini_markov

# embedding the current context
emb = C[torch.tensor([context])]

# pass through the network layers
h = torch.tanh(emb.view(1, -1) @ W1 + b1)
logits = h @ W2 +b2

# compute the probabilities
# squeeze removes unnecessary dimensions
# oliver note: for the curse of dimensionality
probs = F.softmax(logits, dim=1).squeeze()

# print out the porbabilities for each character
next_char_probs = {itos[i]: probs[i].item() 
                   for i in range(len(probs))}
# print(next_char_probs)

# Step 5: Generating New Names

context = [0] * block_size
print("'neural net, start generating pokemony ai names'\nblock size:",block_size,"\n")
for q in range(20): # tutorial uses _ as iter variable but that's bad coding style to me
    out = []
    while True:
        emb = C[torch.tensor([context])]
        h = torch.tanh(emb.view(1, -1) @ W1 +b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix =  torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:
            print(''.join(itos[i] 
                  for i in out))
            break
        else: # moved this to an else statement to get it to stop printing .s and elim blank names
            out.append(ix)

# Things not in this model:
# dynamic learning rate
# overfitting prevention
# expanding context length X
# large dataset
# temperature adjustment