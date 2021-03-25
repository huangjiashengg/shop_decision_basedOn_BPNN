# 第一步，对数据进行预处理，包括数据集划分，归一化等
# 第二步，设定参数，模型训练
# 第三步，可视化结果，包括混淆矩阵
# 第四步，模型调优与可视化结果分析，包括混淆矩阵，模型预测精度随参数调整的变化趋势
# 以上步骤尽量都使用Sklearn进行处理

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
inputfile = 'C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/消费决策数据.txt'
# 读取数据，并去掉表头
with open(inputfile, encoding='utf-8') as f:
    sample = []
    samples = []
    for line in f:
        for l in line.strip().split('\t'):
            sample.append(l)
        samples.append(sample)
        sample = []
samples = samples[1:]
# 转换数据格式为浮点型
new_samples = []
for sample in samples:
    sample = [float(x) for x in sample]
    new_samples.append(sample)
# 分割label以及属性
y = []
X = []
for sample in new_samples:
    y.append(sample[0])
    X.append([sample[1], sample[2], sample[3]])
# print(len(y))  # 共431个样本
# 数据标准化
X = np.array(X)
# print(X)
ss = StandardScaler()
X = ss.fit_transform(X)
# print(X)
# 可视化数据
# plt.figure(figsize=(10, 5))
# plt.subplot(311)
# plt.scatter(range(0, len(X)), X[:, 0])
# plt.subplot(312)
# plt.scatter(range(0, len(X)), X[:, 1])
# plt.subplot(313)
# plt.scatter(range(0, len(X)), X[:, 2])
# plt.show()





# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
# 可视化划分后的数据
# plt.figure(figsize=(10, 5))
# plt.subplot(311)
# plt.scatter(range(0, len(X_train)), X_train[:, 0])
# plt.subplot(312)
# plt.scatter(range(0, len(X_train)), X_train[:, 1])
# plt.subplot(313)
# plt.scatter(range(0, len(X_train)), X_train[:, 2])
# plt.show()

# 一比一随机不放回抽样
# def sampling(len):
#     train_index = random.sample(range(0, len), len // 2)  # 应用random.sample也可以在某数值范围内生成随机数
#     test_index = []
#     for i in range(0, 431):
#         if i not in train_index:
#             test_index.append(i)
#     return train_index, test_index
# # train_index, test_index = sampling(431)
#
#
# # # 训练集与测试集手工划分
# # X_train = [X[x] for x in sampling(len(X))[0]]
# # y_train = [y[x] for x in sampling(len(y))[0]]
# # X_test = [X[x] for x in sampling(len(X))[1]]
# # y_test = [y[x] for x in sampling(len(y))[1]]
# plt.figure()
# plt.plot(X_train, y_train)
# plt.show()
#
# # 模型训练
#
# clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(11, ),
#                     random_state=0, max_iter=1000, learning_rate='adaptive',
#                     verbose=False, activation='logistic')
# # 重要参数：learning_rate学习率、alpha复杂度惩罚系数、优化器、最大迭代次数或者权重的最大调整量小于某个值、激活函数、隐藏层神经元数量及层数；
# # 优化器solver：‘sgd’表示随机梯度下降；‘lbfgs’表示准牛顿方式；‘adam’也是一种随机梯度下降；
# # max_iter最大迭代次数;
# # activation激活函数：logistic表示Sigmoid函数；默认为'relu'---表示[0, x]阶跃函数(小于0则为0大于0则为自己本身，是[0, 1]阶跃的变形)
# # hidden_layer_sizes=(8,)表示只有一层隐藏层，并且神经元数量为8；同理，(8, 4)表示两层，分别为8个神经元和4个神经元
# clf.fit(X_train, y_train)
# # print(clf)
#
# # 模型预测
# predictions = clf.predict(X_test)
# # print(clf.predict_proba(X_test))
# print(predictions)
#
# labels = list(set(y_test))
# conf_mat = confusion_matrix(y_test, predictions, labels=labels)
#
# print("混淆矩阵如下：\n", conf_mat)
# f = clf.score(X_test, y_test, sample_weight=None)*100
# print('预测精度为:%.2f' % f, '%')
#
# 调参，分别对最大迭代次数，惩罚参数以及隐藏神经元进行参数调整
########################################对最大迭代次数调优##########################################################
# accu = []
# for i in range(100, 1000):
#     # 模型训练
#     clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(11,),
#                         random_state=0, max_iter=i, learning_rate='adaptive',
#                         verbose=False, activation='logistic')
#     # 重要参数：learning_rate学习率、alpha复杂度惩罚系数、优化器、最大迭代次数或者权重的最大调整量小于某个值、激活函数、隐藏层神经元数量及层数；
#     # 优化器solver：‘sgd’表示随机梯度下降；‘lbfgs’表示准牛顿方式；‘adam’也是一种随机梯度下降；
#     # max_iter最大迭代次数;
#     # activation激活函数：logistic表示Sigmoid函数；默认为'relu'---表示[0, x]阶跃函数(小于0则为0大于0则为自己本身，是[0, 1]阶跃的变形)
#     # hidden_layer_sizes=(8,)表示只有一层隐藏层，并且神经元数量为8；同理，(8, 4)表示两层，分别为8个神经元和4个神经元
#     clf.fit(X_train, y_train)
#
#     # 模型预测
#     predictions = clf.predict(X_test)
#     # print(clf.predict_proba(X_test))
#
#     labels = list(set(y_test))
#     conf_mat = confusion_matrix(y_test, predictions, labels=labels)
#
#     print("混淆矩阵如下：\n", conf_mat)
#     acc = clf.score(X_test, y_test, sample_weight=None) * 100
#     print('预测精度为:%.2f' % acc, '%')
#     accu.append(acc)
#
# plt.figure()
# plt.plot(range(100, 1000), accu)
# plt.suptitle('Accuracy-max_iter')
# plt.xlabel('max_iterNum')
# plt.ylabel('Accuracy%')
# plt.show()

########################################对惩罚参数调优###############################################
# accu = []
# for i in np.arange(0.000001, 0.1, 0.0001):
#     # 模型训练
#     clf = MLPClassifier(solver='lbfgs', alpha=i, hidden_layer_sizes=(11,),
#                         random_state=0, max_iter=925, learning_rate='adaptive',
#                         verbose=False, activation='logistic')
#     # 重要参数：learning_rate学习率、alpha复杂度惩罚系数、优化器、最大迭代次数或者权重的最大调整量小于某个值、激活函数、隐藏层神经元数量及层数；
#     # 优化器solver：‘sgd’表示随机梯度下降；‘lbfgs’表示准牛顿方式；‘adam’也是一种随机梯度下降；
#     # max_iter最大迭代次数;
#     # activation激活函数：logistic表示Sigmoid函数；默认为'relu'---表示[0, x]阶跃函数(小于0则为0大于0则为自己本身，是[0, 1]阶跃的变形)
#     # hidden_layer_sizes=(8,)表示只有一层隐藏层，并且神经元数量为8；同理，(8, 4)表示两层，分别为8个神经元和4个神经元
#     clf.fit(X_train, y_train)
#
#     # 模型预测
#     predictions = clf.predict(X_test)
#     # print(clf.predict_proba(X_test))
#
#     labels = list(set(y_test))
#     conf_mat = confusion_matrix(y_test, predictions, labels=labels)
#
#     print("混淆矩阵如下：\n", conf_mat)
#     acc = clf.score(X_test, y_test, sample_weight=None) * 100
#     print('预测精度为:%.2f' % acc, '%')
#     accu.append(acc)
#
# plt.figure(figsize=(20, 5))
# plt.plot(np.arange(0.000001, 0.1, 0.0001), accu)
# plt.suptitle('Accuracy-alpha')
# plt.xlabel('alpha_val')
# plt.ylabel('Accuracy%')
# plt.show()
#
# accuracy_high = {}
# k = np.arange(0.000001, 0.1, 0.0001)
# for i in range(0, len(accu)):
#     if accu[i] > 62:
#         accuracy_high[k[i]] = accu[i]
# sort_accuracy_high = sorted(accuracy_high.items(), key=lambda item: item[1], reverse=True)
# print(sort_accuracy_high)

########################第三个参数调优：隐藏层及其节点数############################################
accu = []
for i in np.arange(3, 150, 1):
    # 模型训练
    clf = MLPClassifier(solver='lbfgs', alpha=0.0041, hidden_layer_sizes=(i,),
                        random_state=0, max_iter=925, learning_rate='adaptive',
                        verbose=False, activation='logistic')
    # 重要参数：learning_rate学习率、alpha复杂度惩罚系数、优化器、最大迭代次数或者权重的最大调整量小于某个值、激活函数、隐藏层神经元数量及层数；
    # 优化器solver：‘sgd’表示随机梯度下降；‘lbfgs’表示准牛顿方式；‘adam’也是一种随机梯度下降；
    # max_iter最大迭代次数;
    # activation激活函数：logistic表示Sigmoid函数；默认为'relu'---表示[0, x]阶跃函数(小于0则为0大于0则为自己本身，是[0, 1]阶跃的变形)
    # hidden_layer_sizes=(8,)表示只有一层隐藏层，并且神经元数量为8；同理，(8, 4)表示两层，分别为8个神经元和4个神经元
    clf.fit(X_train, y_train)

    # 模型预测
    predictions = clf.predict(X_test)
    # print(clf.predict_proba(X_test))

    labels = list(set(y_test))
    conf_mat = confusion_matrix(y_test, predictions, labels=labels)

    print("混淆矩阵如下：\n", conf_mat)
    acc = clf.score(X_test, y_test, sample_weight=None) * 100
    print('预测精度为:%.2f' % acc, '%')
    accu.append(acc)

plt.figure(figsize=(20, 5))
plt.plot(np.arange(3, 150, 1), accu)
plt.suptitle('Accuracy-hidden_Layer_NUM')
plt.xlabel('hidden_Layer_NUM')
plt.ylabel('Accuracy%')
plt.show()

accuracy_high = {}
k = np.arange(3, 150, 1)
for i in range(0, len(accu)):
    if accu[i] > 62:
        accuracy_high[k[i]] = accu[i]
sort_accuracy_high = sorted(accuracy_high.items(), key=lambda item: item[1], reverse=True)
print(sort_accuracy_high)

















