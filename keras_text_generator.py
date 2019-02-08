# -*- coding: utf-8 -*-
"""
Created on Fri Feb  8 10:44:34 2019

@author: kagitani
"""

###############################################
###########LSTMによるテキスト生成##################
###############################################

import numpy as np

#original_distributionは確率値からなる一次元の配列
#temperatureは出力分布のエントロピーを定量化する係数
#temperatureが大きいほどランダム性が大きく一様分布になる
def reweight_distribution(original_distribution, temperature=0.5):
    distribution = np.log(original_distribution) / temperature
    distribution = np.exp(distribution)
    
    return distribution / np.sum(distribution)

import keras
import os

os.chdir("C:\\Users\\kagitani\\Documents\\keras_DL")
path = keras.utils.get_file(
        'nietzche.txt',
        origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')

text = open(path, encoding='utf-8').read().lower()   
print('Corpus length:', len(text))

#文字のシーケンスのベクトル化
maxlen = 60      #60文字のシーケンスを抽出
step = 3         #3文字おきに新しいシーケンスをサンプリング
sentences = []   #抽出されたシーケンスを保持
next_chars = []  #次に来る文字を保持

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

print('Number of sequences:', len(sentences))

#set() : 重複した要素を取り除く
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))

char_indices = dict((char, chars.index(char)) for char in chars)

print('Vectorization...')

#one-hotエンコーディングを適応して文字を二値の配列に格納
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
        
#次の文字を予測する単層LSTMモデル
from keras import layers

model = keras.models.Sequential()
model.add(layers.LSTM(128, input_shape=(maxlen, len(chars))))
model.add(layers.Dense(len(chars), activation='softmax'))

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#モデルの予測に基づいて次の文字をサンプリングする関数
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

import random
import sys

for epoch in range(1,60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    
    #テキストシートをランダムに選択
    start_index = random.randint(0, len(text) - maxlen - 1)
    generated_text = text[start_index : start_index + maxlen]
    print('------- Generating with seed: "' + generated_text + '"')
    #ある範囲内の異なるサンプリング温度を試してみる
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------------ temperature:', temperature)
        sys.stdout.write(generated_text)
        
        #400文字生成
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            
            preds = model.predict(sampled, verbose=0)[0]
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            
            generated_text += next_char
            generated_text = generated_text[1:]
            
            sys.stdout.write(next_char)
            sys.stdout.flush()
        
        