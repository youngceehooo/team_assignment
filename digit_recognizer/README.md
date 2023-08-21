# digit_recognizer
## 模型使用说明
get_ipython().run_line_magic('matplotlib', 'inline')该语句调用ipython魔法命令，不确定能否以.py文件形式运行，如果报错可以删掉\
df = pd.read_csv("C:\\Users\\obvious\\Desktop\\digit_recognizer\\train.csv")读取训练数据集，注意修改地址\
df_test = pd.read_csv("C:\\Users\\obvious\\Desktop\\digit_recognizer\\test.csv")读取测试数据集\
res_df.to_csv('res.csv',index=False)导出输出结果\
## 模型效果
pytorch版本kaggle得分在0.98左右


```python

```
