#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 导入相应的工具包
import numpy as np
from matplotlib import pyplot as plt

# tf 中使用工具包
import tensorflow as tf
# 数据集
from tensorflow.keras.datasets import mnist
# 构建模型
from tensorflow.keras.models import Sequential
# 导入需要的层
from tensorflow.keras.layers import Dense, Dropout, Activation, BatchNormalization
# 导入辅助工具包
from tensorflow.keras import utils
# 正则化
from tensorflow.keras import regularizers


# In[2]:


# 数据集中的类别总数
nb_classes = 10
# 加载数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# In[3]:


X_train.shape


# In[4]:


X_test.shape


# In[5]:


# 显示数据
plt.figure()
plt.rcParams['figure.figsize'] = (7, 7)
plt.imshow(X_train[1], cmap='gray')


# In[6]:


# 调整数据维度: 每一个数字转换成一个向量
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
# 格式转换
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# 归一化
X_train /= 255
X_test /= 255
# 维度调整后的结果
print(f'训练集: {X_train.shape}')
print(f'测试集: {X_test.shape}')


# In[75]:


# 将目标值转换成独热编码的形式
y_train = utils.to_categorical(y_train, nb_classes)
y_test = utils.to_categorical(y_test, nb_classes)
print(y_train)


# In[8]:


# 利用序列模型来构建模型
model = Sequential()
# 全连接层, 共 512 个神经元, 输入维度大小为 784
model.add(tf.keras.Input(shape=(784,)))
model.add(Dense(512))
# 激活函数使用 relu
model.add(Activation('relu'))
# 使用正则化方法 dropout
model.add(Dropout(0.2))
#全连接层 512个神经元 加入 L2 正则化
model.add(Dense(512, kernel_regularizer = regularizers.l2(0.001)))
# BN 层
model.add(BatchNormalization())
# 激活函数
model.add(Activation('relu'))
model.add(Dropout(0.2))
# 输出层 共 10 个神经元
model.add(Dense(10))
# softmax 将神经网络的输出的 score 转换为概率值
model.add(Activation('softmax'))


# In[9]:


# 模型编译 指明损失函数和优化器 评估指标
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])


# In[10]:


# batch_size 是每次送入模型的样本个数, epochs 是所有样本的迭代次数, 并指明验证数据集
history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_data=(X_test, y_test))


# In[11]:


# 绘制损失函数的变化曲线
plt.figure()
# 训练集损失函数变化
plt.plot(history.history['loss'], label='train_loss')
# 验证集损失函数变化
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
# 绘制网格
plt.grid()


# In[12]:


# 绘制准确率的变化曲线
plt.figure()
# 训练集损失函数变化
plt.plot(history.history['accuracy'], label='train_accuracy')
# 验证集损失函数变化
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.grid()


# In[18]:


# 模型测试
score = model.evaluate(X_test, y_test, verbose=1)
# 打印结果
print(f'测试集准确率: {score}')


# In[52]:


import pandas as pd
pred_X = pd.read_csv("C:\\Users\\obvious\\Desktop\\digit_recognizer\\test.csv")
predictions=model.predict(pred_X)
predictionsDf=pd.DataFrame(predictions)
pred_X_size=pred_X.shape[0]
num = [i for i in range(1,pred_X_size+1)]
predictionsDf=pd.DataFrame(predictions)
predictionsDf.to_csv('手写数字识别预测结果.csv',index=True)


# In[53]:


predictionsDf.head


# In[57]:


predictionsDf


# In[74]:


df = pd.DataFrame()  
  
# 生成28000个包含0-9的列  
for i in range(28000):  
    column = pd.Series(range(10))  
    df[i] = column  
  
# 查看前5行数据  
print(df)


# In[ ]:


result = predictionsDf.dot(df)


# In[ ]:




