# 泰坦尼克幸存者预测
## 使用方法
在train = pd.read_csv("C:\\Users\\obvious\\Desktop\\titanic\\train.csv")中放入训练数据集路径\
在test = pd.read_csv("C:\\Users\\obvious\\Desktop\\titanic\\test.csv")中放入测试数据集路径\
在predictionsDf.to_csv('predictions_1.csv',index=True)语句中指定输出预测结果的路径\
## 本文件夹中文件内容
predictions_1.csv为预测结果文件\
train.csv为训练集\
test.csv为测试集\
泰坦尼克号幸存者预测.py是由.ipynb文件转换的.py文件\
gender_submission.csv为kaggle提交规范
## 模型效果
本模型是我们的第一个模型，借鉴了csdn上的代码，在原代码的基础上加入了对年龄要素的分类，但因能力有限只是粗略的划分，将原本0.72的kaggle得分提高到0.78，仍存在较大的提升空间



```python

```
