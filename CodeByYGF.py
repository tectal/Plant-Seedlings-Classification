# -*- coding: utf-8 -*-  
# @Time    : 2018/7/4 16:10
# @Author  : Yang Guofeng
# @File    : Code.py  
# @Software: PyCharm 2018.1 (Professional Edition) 




###########################################################################
# 1 加载数据
###########################################################################
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from glob import glob
import cv2

# 定义网络参数
seed = 1              # 随机种子
dim = 70              # 序列最大长度
batch_size = 32       # 批数据量大小
embedding_dims = 50   # 词向量维度
nb_epoch = 5          # 迭代轮次

# 1.1 输入图片、标签，调整图片尺寸
# trainImg为训练图片集；trainLabel为训练标签集
path = '../input/train/*/*.png'
files = glob(path)
trainImg = []
trainLabel = []
j = 1
num = len(files)
for img in files:
    print(str(j) + "/" + str(num), end="\r")
    trainImg.append(cv2.resize(cv2.imread(img), (dim, dim)))
    trainLabel.append(img.split('\\')[-2])
    j += 1
trainImg = np.asarray(trainImg)
trainLabel = pd.DataFrame(trainLabel)

# train数据集部分图片
for i in range(9):
    plt.subplot(3,3,i+1)
    plt.imshow(trainImg[i])
plt.show()

# 1.2 种类标签
from keras.utils import np_utils
from sklearn import preprocessing
import matplotlib.pyplot as plt

# 编码种类标签，创建分类
le = preprocessing.LabelEncoder()
le.fit(trainLabel[0])
print("Classes: " + str(le.classes_))
encodeTrainLabels = le.transform(trainLabel[0])

# 对标签进行分类
clearTrainLabel = np_utils.to_categorical(encodeTrainLabels)
num_clases = clearTrainLabel.shape[1]
print("Number of classes: " + str(num_clases))

# 数据集分类统计
trainLabel[0].value_counts().plot(kind='bar')
plt.show()



'''
###########################################################################
# 2 数据预处理
###########################################################################
# 2.1 过滤图片非绿色部分
clearTrainImg = []
getEx = True
for img in trainImg:
    # 高斯模糊
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    
    # 将图片转换到HSV颜色空间
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    
    # 创建绿色蒙版
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 图片二值蒙版，过滤非绿色部分
    bMask = mask > 0
    clear = np.zeros_like(img, np.uint8)
    clear[bMask] = img[bMask]
    clearTrainImg.append(clear)
    
    # 图片预处理过程
    if getEx:
        plt.subplot(2, 3, 1); plt.imshow(img)     # 原图
        plt.subplot(2, 3, 2); plt.imshow(blurImg) # 模糊图像
        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV图片
        plt.subplot(2, 3, 4); plt.imshow(mask)    # 绿色蒙版
        plt.subplot(2, 3, 5); plt.imshow(bMask)   # 二值蒙版
        plt.subplot(2, 3, 6); plt.imshow(clear)   # 过滤图片的背景
        getEx = False
        plt.show()

clearTrainImg = np.asarray(clearTrainImg)

# 数据集图片经过处理后的部分照片
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(clearTrainImg[i])
plt.show()

# 2.2 标准化输入
clearTrainImg = clearTrainImg / 255




###########################################################################
# 3 构建模型
###########################################################################
# 3.1 划分train文件夹图片数据集
from sklearn.model_selection import train_test_split

# 采用Segmentation处理方法的图片数据集为clearTrainImg；原始图片数据集trainImg
trainX, testX, trainY, testY = train_test_split(clearTrainImg,clearTrainLabel,test_size=0.1,random_state=seed,stratify = clearTrainLabel)

# 将数据集保存为.npz文件，方便储存和读取
# np.savez("Data.npz", trainX=trainX, testX=testX, trainY=trainY, testY=testY)

# 3.2 图片生成器（数据增强）
from keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        rotation_range=180,     # 图片随机旋转180度
        zoom_range = 0.1,       # 图片随机缩放
        width_shift_range=0.1,  # 图片随机水平偏移
        height_shift_range=0.1, # 图片随机竖直偏移
        horizontal_flip=True,   # 图片随机水平翻转
        vertical_flip=True      # 图片随机竖直翻转
    )  
datagen.fit(trainX)

# 3.3 模型结构
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import BatchNormalization
# from keras.layers.advanced_activations import LeakyReLU, PReLU
np.random.seed(seed)

model = Sequential()

# Input Shape (70, 70, 64) 
# Output Shape (70-5)/1+1=66 (66-5)/1+1=62 (62-2)/2+1=31
model.add(Conv2D(filters=64, kernel_size=(5, 5), input_shape=(dim, dim, 3), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

# Input Shape (31, 31, 64)
# Output Shape (31-5)/1+1=27 (27-5)/1+1=23 (23-2)/2+1=11

# alpha：大于0的浮点数，代表激活函数图像中第三象限线段的斜率
# model.add(Activation(LeakyReLU(alpha=0.01)))

# alpha_initializer：alpha的初始化函数
# alpha_regularizer：alpha的正则项
# alpha_constraint：alpha的约束项
# shared_axes：该参数指定的轴将共享同一组科学系参数，例如假如输入特征图是从2D卷积过来的，具有形如(batch, height, width, channels)这样的shape，则或许你会希望在空域共享参数，这样每个filter就只有一组参数，设定shared_axes=[1,2]可完成该目标
# model.add(PReLU(alpha_initializer='0.01', alpha_regularizer=None, alpha_constraint=None, shared_axes=None))

model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=128, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

# Input Shape (11, 11, 128)
# Output Shape (11-5)/1+1=7 (7-5)/1+1=3 (3-2)/2+1=1
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(BatchNormalization(axis=3))
model.add(Conv2D(filters=256, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(BatchNormalization(axis=3))
model.add(Dropout(0.1))

# Dense layers
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(num_clases, activation='softmax'))

model.summary()

# 编译模型
# 优化函数Adam或SGD
# from keras import optimizers

# Adam 该优化器的默认值来源于参考文献
# lr：大或等于0的浮点数，学习率
# beta_1/beta_2：浮点数， 0<beta<1，通常很接近1
# epsilon：大或等于0的小浮点数，防止除0错误
# adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# SGD 随机梯度下降法，支持动量参数，支持学习衰减率，支持Nesterov动量
# lr：大或等于0的浮点数，学习率
# momentum：大或等于0的浮点数，动量参数
# decay：大或等于0的浮点数，每次更新后的学习率衰减值
# nesterov：布尔值，确定是否使用Nesterov动量
# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 3.4 训练模型
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TensorBoard

# tensorboard可视化
tensorboad = TensorBoard(log_dir='../input/log')

# 当评价指标不在提升时，减少学习率
# 当学习停滞时，减少2倍或10倍的学习率常常能获得较好的效果。该回调函数检测指标的情况，如果在patience(3)个epoch中看不到模型性能提升，则减少学习率
# monitor：被监测的量
# factor：每次减少学习率的因子，学习率将以lr = lr*factor的形式被减少
# patience：当patience个epoch过去而模型性能不提升时，学习率减少的动作会被触发
# mode：‘auto’，‘min’，‘max’之一，在min模式下，如果检测值触发学习率减少。在max模式下，当检测值不再上升则触发学习率减少。
# epsilon：阈值，用来确定是否进入检测值的“平原区”
# cooldown：学习率减少后，会经过cooldown个epoch才重新进行正常操作
# min_lr：学习率的下限
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.4,min_lr=0.00001)

# filepath可以是格式化的字符串，里面的占位符将会被epoch值和传入on_epoch_end的logs关键字所填入
# filename：字符串，保存模型的路径
# monitor：需要监视的值
# verbose：信息展示模式，0或1
# save_best_only：当设置为True时，将只保存在验证集上性能最好的模型
# mode：‘auto’，‘min’，‘max’之一，在save_best_only=True时决定性能最佳模型的评判准则，例如，当监测值为val_acc时，模式应为max，当检测值为val_loss时，模式应为min。在auto模式下，评价准则由被监测值的名字自动推断。
# save_weights_only：若设置为True，则只保存模型权重，否则将保存整个模型（包括模型结构，配置信息等）
# period：CheckPoint之间的间隔的epoch数
# 保存最优模型权重
filepath = "weights.best_{epoch:02d}-{val_acc:.2f}.hdf5"
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True, mode='max')

# 回调（学习率，最优模型权重，最后模型权重）, checkpoint_best
callback_lists = [tensorboad, learning_rate_reduction, checkpoint, checkpoint_last]

# 训练模型，设置不同的epochs进行对比
hist = model.fit(trainX, trainY, batch_size=75,epochs=50,validation_data=(testX, testY),callbacks=callback_lists)




###########################################################################
# 4 模型评估
###########################################################################
# 加载模型权重
from keras.models import load_model
model = load_model("weights.best_17-0.96.hdf5")

# 加载划分后的数据集
data = np.load("../input/Data.npz")
d = dict(zip(("trainX","testX","trainY", "testY"), (data[k] for k in data)))
trainX = d['trainX']
testX = d['testX']
trainY = d['trainY']
testY = d['testY']
print(d["trainX"])

# 评估模型分类准确率
print(model.evaluate(trainX, trainY))
print(model.evaluate(testX, testY))
'''



###########################################################################
# 5 模型预测分类
###########################################################################
from sklearn import preprocessing

path = '../input/test/*.png'
files = glob(path)
testImg = []
testId = []
j = 1
num = len(files)

from keras.models import load_model
model = load_model("weights.best_18-0.96.hdf5")

# 5.1 输入图片、标签，调整图片尺寸
# testImg为测试图片集
for img in files:
    print("Obtain images: " + str(j) + "/" + str(num), end='\r')
    testId.append(img.split('\\')[-1])
    testImg.append(cv2.resize(cv2.imread(img), (dim, dim)))
    j += 1

testImg = np.asarray(testImg)

# test数据集部分图片
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(testImg[i])
plt.show()

# 5.2 过滤图片非绿色部分
clearTestImg = []
getEx = True
for img in testImg:

    # 高斯模糊
    blurImg = cv2.GaussianBlur(img, (5, 5), 0)   
    
    # 将图片转换到HSV颜色空间
    hsvImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2HSV)  
    
    # 创建绿色蒙版
    lower_green = (25, 40, 50)
    upper_green = (75, 255, 255)
    mask = cv2.inRange(hsvImg, lower_green, upper_green)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # 图片二值蒙版，过滤背景
    bMask = mask > 0
    clear = np.zeros_like(img, np.uint8)
    clear[bMask] = img[bMask]
    clearTestImg.append(clear)
    
    # 图片预处理过程
    if getEx:
        plt.subplot(2, 3, 1); plt.imshow(img)     # 原图
        plt.subplot(2, 3, 2); plt.imshow(blurImg) # 模糊图像
        plt.subplot(2, 3, 3); plt.imshow(hsvImg)  # HSV图片
        plt.subplot(2, 3, 4); plt.imshow(mask)    # 绿色蒙版
        plt.subplot(2, 3, 5); plt.imshow(bMask)   # 二值蒙版
        plt.subplot(2, 3, 6); plt.imshow(clear)   # 过滤图片的背景
        getEx = False
        plt.show()

clearTestImg = np.asarray(clearTestImg)

# 数据集图片经过处理后的部分照片
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(clearTestImg[i])
plt.show()

# 5.3 标准化输入
clearTestImg = clearTestImg / 255

# 5.4 对5.3输入图片进行预测分类
pred = model.predict(clearTestImg)

# 5.5 输出实验结果.csv文件
predNum = np.argmax(pred, axis=1)
le = preprocessing.LabelEncoder()
le.fit(trainLabel[0])
predStr = le.classes_[predNum]
res = {'file': testId, 'species': predStr}
res = pd.DataFrame(res)
res.to_csv("sample_submission.csv", index=False)
