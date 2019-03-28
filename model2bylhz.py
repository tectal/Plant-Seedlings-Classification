# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 14:12:53 2018

@author: PEACEMINUSONE
"""

import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from keras.layers import Lambda
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm

INV_CLASS = {
    0: 'Black-grass',
    1: 'Charlock',
    2: 'Cleavers',
    3: 'Common Chickweed',
    4: 'Common wheat',
    5: 'Fat Hen',
    6: 'Loose Silky-bent',
    7: 'Maize',
    8: 'Scentless Mayweed',
    9: 'Shepherds Purse',
    10: 'Small-flowered Cranesbill',
    11: 'Sugar beet'
}

#激活函数
def LRelu(x,leak=0.2,name="LR"):
    with tf.variable_scope(name):
        f1 = 0.5*(1+leak)
        f2 = 0.5*(1-leak)
        return f1*x +f2*tf.abs(x)
#给出参数
BATCH_SIZE = 16
EPOCHS = 50
RANDOM_STATE = 11

species = ['Black-grass', 'Charlock', 'Cleavers', 'Common Chickweed', 'Common wheat', 'Fat Hen',
          'Loose Silky-bent', 'Maize','Scentless Mayweed', 'Shepherds Purse',
          'Small-flowered Cranesbill', 'Sugar beet']
data_dir = '/home/xsl/pltdata/project/'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
# 导入训练集数据
train_data = []
for species_id, sp in enumerate(species):
    for file in os.listdir(os.path.join(train_dir, sp)):
        train_data.append(['train/{}/{}'.format(sp, file), species_id, sp])
        
train = pd.DataFrame(train_data, columns=['File', 'SpeciesId','Species'])
train.head()
# 训练集随机化
SEED = 42
train = train.sample(frac=1, random_state=SEED) 
train.index = np.arange(len(train)) # Reset indices
train.head()
# 导入测试集数据
test_data = []
for file in os.listdir(test_dir):
    test_data.append(['test/{}'.format(file), file])
test = pd.DataFrame(test_data, columns=['Filepath', 'File'])
test.head()

IMAGE_SIZE = 51

def read_image(filepath):
    return cv2.imread(os.path.join(data_dir, filepath)) # Loading a color image is the default flag
# 改变图片大小
def resize_image(image, image_size):
    return cv2.resize(image.copy(), image_size, interpolation=cv2.INTER_AREA)
#将目标从背景中分离
def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask
#returns an image mask: Matrix with shape (image_height, image_width).
#In this matrix there are only 0 and 1 values. The 1 values define the interesting part of the original image. 
def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp
X_train = np.zeros((train.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
for i, file in tqdm(enumerate(train['File'].values)):
    image = read_image(file)
    image = segment_plant(image)
    image_segmented = sharpen_image(image)
    X_train[i] = resize_image(image_segmented, (IMAGE_SIZE, IMAGE_SIZE))
# 数据标准化
X_train = X_train / 255.
print('Train Shape: {}'.format(X_train.shape))
Y_train = train['SpeciesId'].values
Y_train = to_categorical(Y_train, num_classes=12)
#同样地处理用于测试集
X_test = np.zeros((test.shape[0], IMAGE_SIZE, IMAGE_SIZE, 3))
label = []
for i, file in tqdm(enumerate(test['Filepath'].values)):
    image = read_image(file)
    image = segment_plant(image)
    image_segmented = sharpen_image(image)
    X_test[i] = resize_image(image_segmented, (IMAGE_SIZE, IMAGE_SIZE))
    label = test['File'].values
X_test = X_test / 255.
# 定义全连接层
def dense_set(inp_layer, n, activation, drop_rate=0.):
    dp = Dropout(drop_rate)(inp_layer)
    dns = Dense(n)(dp)
    bn = BatchNormalization(axis=-1)(dns)
    act = Activation(activation=activation)(bn)
    return act

# 定义卷积层
def conv_layer(feature_batch, feature_map, kernel_size=(3, 3),strides=(1,1)):
    zp = feature_batch
    conv = Conv2D(filters=feature_map, kernel_size=kernel_size, strides=strides)(zp)
    bn = BatchNormalization(axis=3)(conv)
    act = Lambda(lambda bn:LRelu(bn))(bn)
    return act

# 设计网络
def get_model():
    inp_img = Input(shape=(51, 51, 3))

    # 51
    conv1 = conv_layer(inp_img, 64)
    conv2 = conv_layer(conv1, 64)
    mp1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv2)
    # 23 计算：（51-3）/1+1=49 (49-3)/1+1=47 (47-3)/2+1=23
    conv3 = conv_layer(mp1, 128)
    conv4 = conv_layer(conv3, 128)
    mp2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv4)
    # 9
    conv7 = conv_layer(mp2, 256)
    conv8 = conv_layer(conv7, 256)
    conv9 = conv_layer(conv8, 256)
    mp3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv9)
    # 1
    # dense layers
    flt = Flatten()(mp3)
    ds1 = dense_set(flt, 128, activation='tanh')
    out = dense_set(ds1, 12, activation='softmax')

    model = Model(inputs=inp_img, outputs=out)
    
    #优化函数Adam或SGD
    
    mypotim = Adam(lr=2 * 1e-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    #mypotim = SGD(lr=1 * 1e-1, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy',
                   optimizer=mypotim,
                   metrics=['accuracy'])
    model.summary()
    return model


def get_callbacks(filepath, patience=5):
    lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, epsilon=1e-5, patience=patience, verbose=1)
    #monitor：被监测的量 factor:每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少 
    #patience:当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
    #epsilon：阈值，用来确定是否进入检测值的“平原区”
    msave = ModelCheckpoint(filepath, save_best_only=True)
    #在每个epoch后保存模型到filepath save_best_only=True只保存在验证集上性能最好的模型
    return [lr_reduce, msave]

# train model
def train_model(img, target):
    callbacks = get_callbacks(filepath='/home/xsl/pltdata/project/weight/save_weight.hdf5', patience=6)
    gmodel = get_model()
    gmodel.load_weights(filepath='/home/xsl/pltdata/project/weight/give_weight.hdf5')
    x_train, x_valid, y_train, y_valid = train_test_split(
                                                        img,
                                                        target,
                                                        shuffle=True,
                                                        train_size=0.8,
                                                        random_state=RANDOM_STATE
                                                        )
    gen = ImageDataGenerator(
            rotation_range=360.,
            width_shift_range=0.3,
            height_shift_range=0.3,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=True
    )
    #用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。
    #rotation_range：整数，数据提升时图片随机转动的角度
    #width_shift_range：浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
    #height_shift_range：浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
    #zoom_range：浮点数或形如[lower,upper]的列表，随机缩放的幅度，若为浮点数，则相当于[lower,upper] = [1 - zoom_range, 1+zoom_range]
    #horizontal_flip：布尔值，进行随机水平翻转
    #vertical_flip：布尔值，进行随机竖直翻转
    gmodel.fit_generator(gen.flow(x_train, y_train,batch_size=BATCH_SIZE),
               steps_per_epoch=10*len(x_train)/BATCH_SIZE,
               epochs=EPOCHS,
               verbose=1,
               shuffle=True,
               validation_data=(x_valid, y_valid),
               callbacks=callbacks)
    #逐个生成数据的batch并进行训练，降低对显存的占用。
    #steps_per_epoch：整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
#test model
def test_model(img, label):
    gmodel = get_model()
    gmodel.load_weights(filepath='/home/xsl/pltdata/project/weight/save_weight.hdf5')
    prob = gmodel.predict(img, verbose=1)
    pred = prob.argmax(axis=-1)
    sub = pd.DataFrame({"file": label,
                         "species": [INV_CLASS[p] for p in pred]})
    sub.to_csv("sub.csv", index=False, header=True)

# 开始训练
def main():
    train_model(X_train, Y_train)
    test_model(X_test, label)

if __name__=='__main__':
    main()







