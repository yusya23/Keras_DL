# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 09:52:16 2019

@author: kagitani
"""
#単語のone-hotエンコーディング
import numpy as np

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
token_index = {}

for sample in samples:
    for word in sample.split():
        if word not in token_index:
            token_index[word] = len(token_index) + 1
#結果の格納場所
max_length = 10
results = np.zeros((len(samples),
                    max_length,
                    max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i, j, index] = 1
        
#文字レベルでのone-hotエンコーディング
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']
characters = string.printable
token_index = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples),
                    max_length,
                    max(token_index.values()) + 1))

for i, sample in enumerate(samples):
    for j, character in enumerate(sample[:max_length]):
        index = token_index.get(character)
        results[i, j, index] = 1.

#Kerasを用いた単語レベルのone-hotエンコーディング
from keras.preprocessing.text import Tokenizer

sample = ['The cat sat on the mat.', 'The dog ate my homework.']
tokenizer = Tokenizer(num_words=1000)

tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)

one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#単語埋め込みを用いた学習
from keras.layers import Embedding

embedding_layer = Embedding(1000, 64)

from keras.datasets import imdb
from keras import preprocessing

max_features = 10000
#レヴューの最初の20個の単語で分類
max_len = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Embedding

model = Sequential()
model.add(Embedding(10000, 8, input_length=max_len))
#(sample, max_len*8)の形状に変形
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
model.summary()

history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)

#学習済みの単語埋め込み
import os
import pandas as pd
import codecs

os.chdir("C:\\Users\\kagitani\\Documents\\keras_DL")
#Imdb_data = pd.read_csv('imdb_master.csv', encoding='shift-jis')
#文字コードエラーが起こるので以下を使用
with codecs.open("imdb_master.csv", "r", "Shift-JIS", "ignore") as file:
    df = pd.read_table(file, delimiter=",")
    
    

