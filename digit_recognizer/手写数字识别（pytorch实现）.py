#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import nn
get_ipython().run_line_magic('matplotlib', 'inline')

#读取数据
df = pd.read_csv("C:\\Users\\obvious\\Desktop\\digit_recognizer\\train.csv")
#从数据中将标签取出，然后将数据类型转为tensor
labels=torch.from_numpy(np.array(df['label']))
#转为one-hot
ones = torch.sparse.torch.eye(10)
label_one_hot=ones.index_select(0,labels)
#将标签给从数据集中删除，得到训练集的输入数据
df.pop('label')
#将训练集转为tensor
imgs=torch.from_numpy(np.array(df))
imgs = imgs.to(torch.float32)

#相关参数
train_batch_size=64
num_epoches=20
lr=0.01

#构造一个Pytorch数据迭代器
def load_array(data_arrays,batch_size,is_train=True):
    #加星号说明为元组
    #TensorDataset 可以用来对 tensor 进行打包
    dataset=data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset,batch_size,shuffle=is_train)
train_data=load_array((imgs,label_one_hot), batch_size=train_batch_size)

#模型搭建
class Net(nn.Module):
    def __init__(self,in_dim,n_hidden_1,n_hidden_2,out_dim):
        super(Net,self).__init__()
        self.layer1=nn.Sequential(nn.Linear(in_dim,n_hidden_1),nn.BatchNorm1d(n_hidden_1))
        self.layer2=nn.Sequential(nn.Linear(n_hidden_1,n_hidden_2),nn.BatchNorm1d(n_hidden_2))
        self.layer3=nn.Sequential(nn.Linear(n_hidden_2,out_dim))
    def forward(self,x):
        x=F.relu(self.layer1(x))
        x=F.relu(self.layer2(x))
        x=self.layer3(x)
        #x=F.softmax(self.layer3(x))
        return x
    
#检测是否有可用的设备
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#实例化网络
model=Net(784,300,100,10)
model.to(device)

#损失函数和优化器
criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=lr,betas=(0.9,0.99))

#训练网络
losses=[]
acces=[]

for epoch in range(num_epoches):
    train_loss=0
    train_acc=0
    model.train()
    #动态修改学习率
   # if epoch%5==0:
    #    optimizer.param_groups[0]['lr']*=0.1
    for img,label in train_data:
        img=img.to(device)
        label=label.to(device)
        
        out=model(img)
        loss=criterion(out,label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss+=loss.item()
        
        _,pred=out.max(1)
        l,pred1=label.max(1)
        num_correct=(pred==pred1).sum().item()
        acc=num_correct/img.shape[0]
        train_acc+=acc
    losses.append(train_loss/len(train_data))
    acces.append(train_acc/len(train_data))
    print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch,train_loss/len(train_data),train_acc/len(train_data)))


# In[11]:


#读取测试集
df_test = pd.read_csv("C:\\Users\\obvious\\Desktop\\digit_recognizer\\test.csv")
test_imgs=torch.from_numpy(np.array(df_test))
test_imgs = test_imgs.to(torch.float32)

#测试集预测部分
#传入GPU
test_imgs=test_imgs.to(device)
#计算结果
_,pre=model(test_imgs).max(1)
print(pre)
#将结果转为提交kaggle的格式
res={}
pre = pre.cpu().numpy()
pre_size=pre.shape[0]
num = [i for i in range(1,pre_size+1)]
res_df=pd.DataFrame({
    'ImageId':num,
    'Label':pre
})

#d导出为CSV文件
res_df.to_csv('res.csv',index=False)


# In[ ]:




