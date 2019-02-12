# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 20:28:42 2019

@author: u5326
"""
#################################################
#############中間層の内容を可視化##################
#################################################

from keras.models import load_model
import os
from keras.preprocessing import image
import numpy as np

os.chdir("C:\\Users\\u5326\\OneDrive\\ドキュメント\\Keras_DL-master\\dogs_vs_cats\\cats_and_dogs_small")
img_path = "C:\\Users\\u5326\\OneDrive\\ドキュメント\\Keras_DL-master\\dogs_vs_cats\\cats_and_dogs_small\\test\\cats\\cat.1700.jpg"
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

img = image.load_img(img_path, target_size=(150, 150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)

img_tensor /= 255.

print(img_tensor.shape)

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

#一層目の活性化を可視化
from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(img_tensor)

import matplotlib.pyplot as plt

first_layer_activation = activations[0]
plt.matshow(first_layer_activation[0, :, :, 10], cmap='viridis')
plt.show()

#中間層の活性化ごとにすべてのチャネルを可視化
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)
images_per_row=16

for layer_name, layer_activation in zip(layer_names, activations):
    
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            
            display_grid[col * size : (col + 1) * size,
                     row * size : (row + 1) * size] = channel_image
                     
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
            
plt.show()
            
#############################################
############# CNNのフィルタの可視化 #############
#############################################

from keras.applications import VGG16
from keras import backend as K
import numpy as np 

model = VGG16(weights='imagenet', include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

#layer_output = model.get_layer(layer_name).output
#loss = K.mean(layer_output[:, :, :, filter_index])
##入力に関する損失関数の勾配を取得
#grads = K.gradients(loss, model.input)[0]
##勾配の正規化
#grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
#
##入力値をNumpy配列で受け取り、出力値をNumpy配列（損失値、勾配値)で返す
#iterate = K.function([model.input], [loss, grads])
#
#import numpy as np
#loss_value, grads_value = iterate([np.zeros((1, 150, 150, 3))])
#
##確率的勾配降下法を使って損失値を最大化
#input_img_data = np.random.random((1, 150, 150, 3)) * 20 + 128.
##勾配上昇法を40ステップ繰り返す
#step = 1.
#for i in range(40):
#    loss_value, grads_value = iterate([input_img_data])
#    input_img_data += grads_value * step
    
    
def deprocess_image(x):
    #テンソルを正規化
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    
    x += 0.5
    x = np.clip(x, 0, 1)
    
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def generate_pattern(layer_name, filter_index, size=150):
    #ターゲット層のn番目のフィルタの活性化を最大化する損失関数を構築
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    #この損失関数を使って入力画像の勾配を計算
    grads = K.gradients(loss, model.input)[0]
    #勾配を正規化
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    #入力画像に基づいて損失値と勾配値を返す
    iterate = K.function([model.input], [loss, grads])
    #最初はノイズが含まれたグレースケールを使用
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    
    step = 1.
    
    for i in range(40):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    return deprocess_image(img)

import matplotlib.pyplot as plt
plt.imshow(generate_pattern(layer_name, 0))
plt.show()

layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
for layer_name in layers:
    size = 64
    margin = 5
    #結果を格納する空画像
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))
    
    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start : horizontal_end,
                    vertical_start : vertical_end, :] = filter_img
    
plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()


############################################################
############# クラスの活性化をヒートマップとして可視化 ###############
############################################################

from keras.applications.vgg16 import VGG16

model = VGG16(weights='imagenet')

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
import numpy as np

img_path = 'C:\\Users\\u5326\OneDrive\\ドキュメント\\Keras_DL-master\\elephant.jpg'

img = image.load_img(img_path, target_size=(224, 224))

x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

x = preprocess_input(x)

preds = model.predict(x)
print('Predicted : ', decode_predictions(preds, top=3)[0])

african_elephant_output = model.output[:, 386]

last_conv_layer = model.get_layer('block5_conv3')

grads = K.gradients(african_elephant_output, last_conv_layer.output)[0]

pooled_grads = K.mean(grads, axis=(0, 1, 2))

iterate = K.function([model.input],
                     [pooled_grads, last_conv_layer.output[0]])

pooled_grads_value, conv_layer_output_value = iterate([x])

for i in range(512):
    conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
    
    heatmap = np.mean(conv_layer_output_value, axis=-1)

heatmap = np.maximum(heatmap, 0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)

import cv2
import os
os.chdir('C:\\Users\\u5326\\OneDrive\\ドキュメント\\Keras_DL-master')
img = cv2.imread('elephant.jpg')

heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

heatmap = np.uint8(255 * heatmap)

heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

superimposed_img = heatmap * 0.4 + img

cv2.imwrite('elephant_cam.jpg', superimposed_img)