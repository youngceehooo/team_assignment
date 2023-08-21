#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv
#导入数据
#训练数据集
train = pd.read_csv("C:\\Users\\obvious\\Desktop\\titanic\\train.csv")
#测试数据集
test  = pd.read_csv("C:\\Users\\obvious\\Desktop\\titanic\\test.csv")
#这里要记住训练数据集有891条数据，方便后面从中拆分出测试数据集用于提交Kaggle结果
print ('训练数据集:',train.shape,'测试数据集:',test.shape)
rowNum_train=train.shape[0]
rowNum_test=test.shape[0]
print('kaggle训练数据集有多少行数据：',rowNum_train,
     ',kaggle测试数据集有多少行数据：',rowNum_test,)
#合并数据集，方便同时对两个数据集进行清洗
full = train.append( test , ignore_index = True )#使用append进行纵向堆叠

print ('合并后的数据集:',full.shape)
# print('处理前：')
print(full.isnull().sum())
#年龄(Age)
full['Age']=full['Age'].fillna( full['Age'].mean() )
#船票价格(Fare)
full['Fare'] = full['Fare'].fillna( full['Fare'].mean() )
# print('处理后：')
print(full.isnull().sum())
full['Embarked'] = full['Embarked'].fillna( 'S' )
full['Cabin'] = full['Cabin'].fillna( 'U' )
sex_mapDict={'male':1,
            'female':0}
#map函数：对Series每个数据应用自定义的函数计算
full['Sex']=full['Sex'].map(sex_mapDict)
embarkedDf = pd.DataFrame()
embarkedDf = pd.get_dummies( full['Embarked'] , prefix='Embarked' )
full = pd.concat([full,embarkedDf],axis=1)
full.drop('Embarked',axis=1,inplace=True)
pclassDf = pd.DataFrame()

#使用get_dummies进行one-hot编码，列名前缀是Pclass
pclassDf = pd.get_dummies( full['Pclass'] , prefix='Pclass' )
full = pd.concat([full,pclassDf],axis=1)

#删掉客舱等级（Pclass）这一列
full.drop('Pclass',axis=1,inplace=True)
def getTitle(name):
    str1=name.split( ',' )[1] #Mr. Owen Harris
    str2=str1.split( '.' )[0]#Mr
    #strip() 方法用于移除字符串头尾指定的字符（默认为空格）
    str3=str2.strip()
    return str3

titleDf = pd.DataFrame()
#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = full['Name'].map(getTitle)
title_mapDict = {
                    "Capt":       "Officer",
                    "Col":        "Officer",
                    "Major":      "Officer",
                    "Jonkheer":   "Royalty",
                    "Don":        "Royalty",
                    "Sir" :       "Royalty",
                    "Dr":         "Officer",
                    "Rev":        "Officer",
                    "the Countess":"Royalty",
                    "Dona":       "Royalty",
                    "Mme":        "Mrs",
                    "Mlle":       "Miss",
                    "Ms":         "Mrs",
                    "Mr" :        "Mr",
                    "Mrs" :       "Mrs",
                    "Miss" :      "Miss",
                    "Master" :    "Master",
                    "Lady" :      "Royalty"
                    }

#map函数：对Series每个数据应用自定义的函数计算
titleDf['Title'] = titleDf['Title'].map(title_mapDict)

#使用get_dummies进行one-hot编码
titleDf = pd.get_dummies(titleDf['Title'])
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,titleDf],axis=1)

#删掉姓名这一列
full.drop('Name',axis=1,inplace=True)
#存放客舱号信息
cabinDf = pd.DataFrame()

'''
客场号的类别值是首字母，例如：
C85 类别映射为首字母C
'''
full[ 'Cabin' ] = full[ 'Cabin' ].map( lambda c : c[0] )#客舱号的首字母代表处于哪个，U代表不知道属于哪个船舱

##使用get_dummies进行one-hot编码，列名前缀是Cabin
cabinDf = pd.get_dummies( full['Cabin'] , prefix = 'Cabin' )
#添加one-hot编码产生的虚拟变量（dummy variables）到泰坦尼克号数据集full
full = pd.concat([full,cabinDf],axis=1)

#删掉客舱号这一列
full.drop('Cabin',axis=1,inplace=True)
familyDf = pd.DataFrame()
familyDf[ 'FamilySize' ] = full[ 'Parch' ] + full[ 'SibSp' ] + 1
familyDf[ 'Family_Single' ] = familyDf[ 'FamilySize' ].map( lambda s : 1 if s == 1 else 0 )
familyDf[ 'Family_Small' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 2 <= s <= 4 else 0 )
familyDf[ 'Family_Large' ]  = familyDf[ 'FamilySize' ].map( lambda s : 1 if 5 <= s else 0 )
full = pd.concat([full,familyDf],axis=1)
full.drop('FamilySize',axis=1,inplace=True)
corrDf = full.corr()
corrDf['Survived'].sort_values(ascending =False)#相关系数法进行简单分析
print(corrDf['Survived'])
#选择相关性较强的标签放入数据框
full_X = pd.concat( [titleDf,#头衔
                     pclassDf,#客舱等级
                     familyDf,#家庭大小
                     full['Fare'],#船票价格
                     full['Sex'],#性别
                     cabinDf,#船舱号
                     embarkedDf,#登船港口
                    ] , axis=1 )
sourceRow=891
#原始数据集：特征
source_X = full_X.loc[0:sourceRow-1,:]
#原始数据集：标签
source_y = full.loc[0:sourceRow-1,'Survived']
#预测数据集：特征
pred_X = full_X.loc[sourceRow:,:]
#建立模型用的训练数据集和测试数据集
from sklearn.model_selection import train_test_split

train_X, test_X, train_y, test_y = train_test_split(source_X ,
                                                        source_y,
                                                      train_size=0.8,
                                                        random_state=5)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=500)
model.fit( train_X , train_y )
predictions=model.predict(pred_X)
predictionsDf=pd.DataFrame(predictions)
predictionsDf.to_csv('predictions_1.csv',index=True)


# In[ ]:




