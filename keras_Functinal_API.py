# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 13:46:55 2019

@author: kagitani
"""
############################################
############## 多入力モデル  #################
############################################

#今回はテキストデータ（Imdb）と数値データ（適当）を入力として一つの日分類結果を出力する。
#画像データ、テキストデータ、数値データが混合しているデータセットに有効。

#############################データ整形#####################################

import os
import pandas as pd
import codecs
import numpy as np

os.chdir("C:\\Users\\kagitani\\Documents\\keras_DL")
#Imdb_data = pd.read_csv('imdb_master.csv', encoding='shift-jis')
#文字コードエラーが起こるので以下を使用
with codecs.open("imdb_master.csv", "r", "Shift-JIS", "ignore") as file:
    df = pd.read_table(file, delimiter=",")
#ラベルを0,1表記に
labels = []
for i in df['label']:
    if i == 'pos':
        labels.append(1)
    elif i == 'neg':
        labels.append(0)
    else:
        labels.append(2)
#0,1表記でないものを削除
df['label_0_1'] = pd.DataFrame(labels)
df = df.drop(df[df['label_0_1']==2].index)
df = df.drop(['Unnamed: 0', 'label', 'file', 'type'], axis=1)
#ランダムの値作成
pos = np.random.randint(51, 150,
                        size=(25000,300))
neg = np.random.randint(1, 100,
                        size=(25000,300))
label = np.vstack((neg,pos))
label = (label - label.mean())/label.std()
label = pd.DataFrame(label)
df_label = pd.concat([df, label], axis=1)

#データシャッフル
rand = np.random.permutation(len(df))
df_rand = df_label.iloc[rand]

labels = np.array(df_rand['label_0_1'][:2500])
#one-hot表記に変更
labels = pd.get_dummies(labels)
texts = np.array(df_rand['review'][:2500])
values_ = np.array(df_rand.iloc[:, 2:][:2500])

########################テキストデータを二次元配列の形に変形#########################

from keras.preprocessing.text import Tokenizer

max_words = 1000            #データセットの出現頻度順の1,000ワードのみを考慮  

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
one_hot_results = tokenizer.texts_to_matrix(texts, mode='binary')

################################多入力モデルの作成#############################

from keras.models import Model
from keras import layers
from keras import Input

text_vocabulary_size = 1000
answer_vocabulary_size = 2

#1.テキストデータのネットワーク構築
#テキスト入力は整数の可変長のシーケンス
text_input = Input(shape=(None,), dtype='float32', name='text')
dense = layers.Embedding(text_vocabulary_size, 64)
#入力をサイズは64のベクトルシーケンスに埋め込む
embedded_text = dense(text_input)
#LSTMを通じてこれらのベクトルを単一のベクトルにエンコーディング
encoded_text = layers.LSTM(32)(embedded_text)

#2.数値データのネットワーク構築
#質問入力でも同じプロセスを繰り返す
value_input = Input(shape=(300,), dtype='float32', name='value')
dense_value_1 = layers.Dense(128, activation='relu')(value_input)
dense_value_2 = layers.Dense(32, activation='relu')(dense_value_1)

#エンコードされたテキストと質問を連結
concatenated = layers.concatenate([encoded_text, dense_value_2], axis=1)
#分類器を追加
answer = layers.Dense(
        answer_vocabulary_size, activation='softmax')(concatenated)

#モデルをインスタンス化するときには、2つの入力と一つの出力を指定
model = Model([text_input, value_input], answer)
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])

#入力ディクショナリを使った適合(入力に名前を付ける場合のみ)
model.fit({'text':one_hot_results, 'value':values_}, labels, epochs=10, batch_size=128)

#from keras.models import Model
#from keras import layers
#from keras import Input
#
#text_vocabulary_size = 10000
#question_vocabulary_size = 10000
#answer_vocabulary_size = 500
#
##テキスト入力は整数の可変長のシーケンス
#text_input = Input(shape=(None,), dtype='int32', name='text')
#
#dense = layers.Embedding(text_vocabulary_size, 64)
#
##入力をサイズは64のベクトルシーケンスに埋め込む
#embedded_text = dense(text_input)
##LSTMを通じてこれらのベクトルを単一のベクトルにエンコーディング
#encoded_text = layers.LSTM(32)(embedded_text)
#
##質問入力でも同じプロセスを繰り返す
#question_input = Input(shape=(None,), dtype='int32', name='question')
#embedded_question = layers.Embedding(
#        question_vocabulary_size, 32)(question_input)
#encoded_question = layers.LSTM(16)(embedded_question)
##エンコードされたテキストと質問を連結
#concatenated = layers.concatenate([encoded_text, encoded_question], axis=1)
##ソフトマックス分類器を追加
#answer = layers.Dense(
#        answer_vocabulary_size, activation='softmax')(concatenated)
#
##モデルをインスタンス化するときには、2つの入力と一つの出力を指定
#model = Model([text_input, question_input], answer)
#model.compile(optimizer='rmsprop',
#              loss='categorical_crossentropy',
#              metrics=['acc'])
#
##多入力モデルへのデータの供給
#import numpy as np
#num_samples = 1000
#max_length = 100
#
##ダミーのNumpyデータを生成
#text = np.random.randint(1, text_vocabulary_size,
#                         size=(num_samples, max_length))
#question = np.random.randint(1, question_vocabulary_size,
#                             size = (num_samples, max_length))
#
##答えにone-hotエンコーディングを適用
#answers = np.zeros(shape=(num_samples, answer_vocabulary_size))
#indices = np.random.randint(0, answer_vocabulary_size, size=num_samples)
#for i, x in enumerate(answers):
#    x[indices[i]] = 1
#
##入力リストを使った適合
#model.fit([text, question], answers, epochs=10, batch_size=128)
##入力ディクショナリを使った適合(入力に名前を付ける場合のみ)
#model.fit({'text':text, 'question':question}, answers, epochs=10, batch_size=128)


#########################################################
####################多出力モデル###########################
#########################################################

from keras import layers
from keras import Input
from keras.models import Model
import numpy as np
import pandas as pd

vocabulary_size = 1000
num_income_groups = 10

posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(32, vocabulary_size)(posts_input)

x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation='relu')(x)
x = layers.Conv1D(128, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation='relu')(x)
x = layers.Conv1D(128, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)

#出力層に名前がついていることに注意
age_prediction = layers.Dense(1, name='age')(x)
income_prediction = layers.Dense(num_income_groups,
                                activation='softmax',
                                name='income')(x)
gender_prediction = layers.Dense(2, activation='sigmoid', name='gender')(x)

model = Model(posts_input,
              [age_prediction, income_prediction, gender_prediction])
model.compile(optimizer='rmsprop',
              loss=['mse',
                    'categorical_crossentropy',
                    'binary_crossentropy'])

#多出力モデルのコンパイル
#出力層に名前を付けている場合に可能
model.compile(optimizer='rmsprop',
              loss={'age':'mse',
                    'income':'categorical_crossentropy',
                    'gender':'binary_crossentropy'})
#多出力モデルのコンパイル(損失の重みづけ)
model.compile(optimizer='rmsprop',
              loss={'age':'mse',
                    'income':'categorical_crossentropy',
                    'gender':'binary_crossentropy'},
              loss_weights={'age' : 0.25, 'income' : 1., 'gender' : 10.})

#ダミーのNumpyデータを生成
age_targets = np.random.randint(1., 80.,
                         size=2500)

income_targets = np.random.randint(1., 11.,
                             size = 2500)
income_targets = np.array(pd.get_dummies(income_targets))

gender_targets = np.random.randint(0., 2.,
                             size = 2500)
gender_targets = np.array(pd.get_dummies(gender_targets))


model.fit(one_hot_results, {'age' : age_targets,
                  'income' : income_targets,
                  'gender' : gender_targets},epochs=10, batch_size=32)