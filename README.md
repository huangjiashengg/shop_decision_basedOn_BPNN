# 购物决策---基于神经网络模型的模型训练与调优

# 前言

本实验是基于神经网络模型的训练与调优记录，旨在通过学习与总结，不存在其他用途与目的。数据采用薛微出版的《R语言数据挖掘方法及应用》教材中提供的数据集---【购物决策数据集】；为简单起见，应用调包SKlearn进行数据预处理与模型训练；后半部分将包含对训练结果的分析、参数调优与最佳模型与参数选择。本实验采用python语言进行模型构建。

模型理解：训练一神经网络模型。得到一个新的数据（至少包含性别年龄收入），只要输入性别、年龄和收入，我们可以预测它是否购买，以及会购买的概率有多大。

## 数据预览

**购物决策数据集如图所示：**

![image-20210325161855113](C:\Users\DELL\PycharmProjects\skearn_prediction\README.assets\image-20210325161855113.png)

**数据说明：**

1，一共包含四个字段；

2，第一个字段为label标签，其他三个字段分别为属性标签：年龄、性别、收入；

## 数据预处理

由于数据存放在‘消费决策数据.txt’文本文件中，每个样本中各个字段之间以‘\t’作为分隔符，因此数据预处理阶段将包含以下几个阶段：

1，读入数据，将样本放入Python列表变量中；

2，将数据由字符型变量转换为浮点型变量；

3，对数据进行标准化或者归一化。

Let's start!

**第一步骤：读入数据，将样本放入Python列表变量中**

引入数据路径

```python
inputfile = 'C:/Users/DELL/Desktop/机器学习/R语言与数据挖掘电子资料/消费决策数据.txt'
```

读取数据，去掉表头

```python
with open(inputfile, encoding='utf-8') as f:
    sample = []
    samples = []
    for line in f:
        for l in line.strip().split('\t'):
            sample.append(l)
        samples.append(sample)
        sample = []
samples = samples[1:]
```

**第二步骤：将数据由字符型变量转换为浮点型变量**

将数据格式转换为浮点型

```python
new_samples = []
for sample in samples:
    sample = [float(x) for x in sample]
    new_samples.append(sample)
```

分割label及属性，方便使用sklearn进行数据集划分

```python
y = []
X = []
for sample in new_samples:
    y.append(sample[0])
    X.append([sample[1], sample[2], sample[3]])
```

**第三步骤：对数据进行标准化**

先引入对应的标准化类

```python
from sklearn.preprocessing import StandardScaler
```

数据标准化

```python
ss = StandardScaler()
X = ss.fit_transform(X)
```

**打印看看标准化前和标准化后的数据对比**

标准化前：                       标准化后：                                              

[[41.  2.  1.]                       [[ 0.35766699  0.89209491 -1.29362056]
  [47.  2.  1.]                        [ 1.41714792  0.89209491 -1.29362056]
  [41.  2.  1.]                        [ 0.35766699  0.89209491 -1.29362056]
  ...                                        ...
  [32.  1.  3.]                        [-1.23155439 -1.12095696  1.16254887]
  [34.  1.  3.]                        [-0.87839408 -1.12095696  1.16254887]
   [34.  1.  3.]]                      [-0.87839408 -1.12095696  1.16254887]]

**对三个属性字段进行可视化**

<img src="C:\Users\DELL\PycharmProjects\skearn_prediction\README.assets\image-20210325164457778.png" alt="image-20210325164457778" style="zoom:50%;" />

第一幅图是Age字段，往下依次是Gender性别和Income收入.

## 数据集分割与格式转换

对数据集进行分割，用Sklearn会非常方便

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=0)
```

我们去X_train进行可视化看看数据分布：

<img src="C:\Users\DELL\PycharmProjects\skearn_prediction\README.assets\image-20210325164853514.png" alt="image-20210325164853514" style="zoom:50%;" />

可以从训练集中第二个字段和第三个字段的数据分布直接看到，对数据集进行划分分割时把顺序打乱了进行随机抽样，分布随机对训练过程比较有利。

## 模型训练

### 模型重要参数选择与设定

在神经网络模型中，最重要的参数无非包括：最大迭代次数，alpha复杂度惩罚参数，优化器solver选择以及隐藏层神经元数量及层数。

**最大迭代次数：**

为了不浪费计算机的计算资源，这里我们将最大迭代次数设置为1000，把学习率设置为0.001

**alpha复杂度惩罚参数：**

alpha=1e-3， 即0.001

**优化器solver：**

选择适用于小数据集的‘lbfgs’(准牛顿迭代方式)

**隐藏层神经元数量及层数：**

暂时只设置一层隐藏层，包含11个神经元hidden_layer_sizes=(11, )

注：同理，(11, 8, )则表示有两层隐藏层，神经元数量分别为11和8.

**代码如下：**

```python
clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(11, ),
                    random_state=0, max_iter=1000, learning_rate='adaptive',
                    verbose=False, activation='logistic')
```

### 用于调优的重要参数

基于对神经网络的原理理解，选择三个对模型影响较大的参数进行模型训练之后的调优，包括：最大迭代次数，惩罚参数以及隐藏神经元进行参数调整。

这里我们先进行模型训练。

### 开始模型训练

```python
clf.fit(X_train, y_train)
```

MLPClassifier(activation='logistic', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(11,), learning_rate='adaptive',
              learning_rate_init=0.001, max_fun=15000, max_iter=1000,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=0, shuffle=True, solver='lbfgs',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)

## 模型评估

1，经过上一步骤的拟合训练，直接调用训练结果对测试集的数据进行预测；

2，输出预测结果，与真实结果进行比对，生成混淆矩阵(此模型为二分类问题)

3，输出模型预测精准度。

```python
# 模型预测
predictions = clf.predict(X_test)
```

**输出结果：**

[0. 1. 1. 1. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 0. 0. 0. 0. 1. 0. 0.
 0. 0. 0. 0. 1. 0. 0. 1. 1. 1. 0. 0. 0. 1. 0. 0. 1. 0. 0. 0. 0. 0. 1. 0.
 1. 0. 1. 0. 1. 0. 1. 1. 1. 0. 0. 0. 0. 0. 0. 1. 1. 0. 0. 1. 0. 0. 1. 0.
 0. 0. 0. 0. 0. 0. 0. 1. 0. 1. 0. 0. 0. 0. 1. 1. 0. 0. 0. 0. 1. 0. 1. 0.
 0. 1. 1. 0. 1. 0. 0. 0. 0. 1. 1. 0.]

```python
labels = list(set(y_test))
conf_mat = confusion_matrix(y_test, predictions, labels=labels)
print("混淆矩阵如下：\n", conf_mat)
f = clf.score(X_test, y_test, sample_weight=None)*100
print('预测精度为:%.2f' % f, '%')
```

**输出结果：**

混淆矩阵如下：
 [[50 17]
 [23 18]]
预测精度为:62.96 %

**结果分析：**

1，预测精度为62.96 %，一个比较bad的结果；

2，回归整个模型创建过程，找出不合理之处：

a. 重要参数设置，最大迭代次数为1000，对于400左右的小样本量，足以让预测结果收敛；

b. 复杂度惩罚参数为0.001，待后续模型参数调优斟酌；

c. 隐藏层只有一层，并且有11个神经元，此处已经将原有3维变量提升到11个维度，待后续模型参数调优斟酌。

## 模型参数调优

前面我们选择了最大迭代次数，惩罚参数以及隐藏神经元进行参数调优。

接下来我们分别针对三个参数，依次进行循环输出相应的模型预测精准度，依次选择最优的该参数进行下一个参数的调优。

**首先，最大迭代次数：**

最大迭代次数我们把范围限定在100到1000之间，看模型何时达到收敛。对最大迭代次数调参，先注释掉以上单个模型训练步骤，其他模型参数保持不变，用for循环改变最大迭代次数，依次输出结果。并用列表变量接收每次训练的结果。

代码如下：

```pyton
accu = []
for i in range(100, 1000):
    # 模型训练
    clf = MLPClassifier(solver='lbfgs', alpha=1e-3, hidden_layer_sizes=(11,),
                        random_state=0, max_iter=i, learning_rate='adaptive',
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

plt.figure()
plt.plot(range(100, 1000), accu)
plt.suptitle('Accuracy-max_iter')
plt.xlabel('max_iterNum')
plt.ylabel('Accuracy%')
plt.show()
```

**部分控制台输出结果：**

混淆矩阵如下：
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
 [[56 11]
 [28 13]]
预测精度为:63.89 %
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
混淆矩阵如下：
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
 [[56 11]

Increase the number of iterations (max_iter) or scale the data as shown in:
 [28 13]]
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
预测精度为:63.89 %...........................................

**图标输出：**

![image-20210325173422610](C:\Users\DELL\PycharmProjects\skearn_prediction\README.assets\image-20210325173422610.png)

由图可看到，在i迭代至i = 900左右时，模型预测精度达到收敛。因此我们把迭代次数设置为925.

**其次，惩罚参数调优：**

惩罚参数我们把调优范围限制在0.000001~0.1之间，间隔0.0001，观察参数在哪个范围内，模型的精准度较高。代码如下：

```python
for i in np.arange(0.000001, 0.1, 0.0001):
    # 模型训练
    clf = MLPClassifier(solver='lbfgs', alpha=i, hidden_layer_sizes=(11,),
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

plt.figure()
plt.plot(np.arange(0.000001, 0.1, 0.0001), accu)
plt.suptitle('Accuracy-alpha')
plt.xlabel('alpha_val')
plt.ylabel('Accuracy%')
plt.show()
```

**控制台部分输出如下：**

混淆矩阵如下：
 [[49 18]
 [26 15]]
预测精度为:59.26 %
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
混淆矩阵如下：
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
 [[51 16]
 [27 14]]
预测精度为:60.19 %

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
混淆矩阵如下：
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
 [[58  9]
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
 [33  8]]
预测精度为:61.11 %

Increase the number of iterations (max_iter) or scale the data as shown in:
    https://scikit-learn.org/stable/modules/preprocessing.html
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
混淆矩阵如下：
 [[48 19]
 [23 18]]
预测精度为:61.11 %
混淆矩阵如下：
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
 [[55 12]
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
 [30 11]]

**图表输出：**

![image-20210325191846876](C:\Users\DELL\PycharmProjects\skearn_prediction\README.assets\image-20210325191846876.png)

分析：

1，这个图表看起来很是起伏不定，难以做抉择；我们把复杂度惩罚和准确率构造成一个字典，输出准确率较高时相应的复杂度惩罚参数；

2，对第一步的输出结果进行排序，选择精准度最高的复杂度惩罚参数作为最终的参数选择。



**输出结果：**

[(0.012901, 69.44444444444444), (0.0041010000000000005, 67.5925925925926), (0.032301, 67.5925925925926), (0.032901, 66.66666666666666), (0.0022010000000000003, 65.74074074074075), (0.007801000000000001, 64.81481481481481), (0.009701, 64.81481481481481), (0.075001, 64.81481481481481), (0.080101, 64.81481481481481), (0.080901, 64.81481481481481), (0.086401, 64.81481481481481), (0.087401, 64.81481481481481), (0.0031010000000000005, 63.888888888888886), (0.008401, 63.888888888888886), (0.013601, 63.888888888888886), (0.016401000000000002, 63.888888888888886), (0.018801000000000002, 63.888888888888886), (0.020401000000000002, 63.888888888888886), (0.021101, 63.888888888888886), (0.022301, 63.888888888888886), (0.022401, 63.888888888888886), (0.022801000000000002, 63.888888888888886), (0.028001, 63.888888888888886), (0.029901000000000004, 63.888888888888886), (0.032201, 63.888888888888886), (0.059301000000000006, 63.888888888888886), (0.000501, 62.96296296296296), (0.001901, 62.96296296296296), (0.0026010000000000004, 62.96296296296296), (0.006601, 62.96296296296296), (0.015101, 62.96296296296296), (0.015401, 62.96296296296296), (0.016101, 62.96296296296296), (0.016201, 62.96296296296296), (0.018301, 62.96296296296296), (0.018501000000000004, 62.96296296296296), (0.020901000000000003, 62.96296296296296), (0.022501000000000004, 62.96296296296296), (0.024901000000000003, 62.96296296296296), (0.027801000000000003, 62.96296296296296), (0.028901000000000003, 62.96296296296296), (0.032601000000000005, 62.96296296296296), (0.035101, 62.96296296296296), (0.039201, 62.96296296296296), (0.039401000000000005, 62.96296296296296), (0.043001000000000004, 62.96296296296296), (0.043301000000000006, 62.96296296296296), (0.045901000000000004, 62.96296296296296), (0.05810100000000001, 62.96296296296296), (0.08070100000000001, 62.96296296296296), (0.081001, 62.96296296296296), (0.0024010000000000004, 62.03703703703704), (0.002801, 62.03703703703704), (0.004501000000000001, 62.03703703703704), (0.005001, 62.03703703703704), (0.0051010000000000005, 62.03703703703704), (0.0064010000000000004, 62.03703703703704), (0.0071010000000000005, 62.03703703703704), (0.007201000000000001, 62.03703703703704), (0.010301, 62.03703703703704), (0.010901, 62.03703703703704), (0.012001, 62.03703703703704), (0.015001, 62.03703703703704), (0.017301, 62.03703703703704), (0.017401000000000003, 62.03703703703704), (0.017501000000000003, 62.03703703703704), (0.017801, 62.03703703703704), (0.018101000000000003, 62.03703703703704), (0.018701000000000002, 62.03703703703704), (0.019401, 62.03703703703704), (0.020301000000000003, 62.03703703703704), (0.022101000000000003, 62.03703703703704), (0.023401, 62.03703703703704), (0.024101, 62.03703703703704), (0.024301000000000003, 62.03703703703704), (0.024401000000000003, 62.03703703703704), (0.027701000000000003, 62.03703703703704), (0.029001000000000002, 62.03703703703704), (0.031601000000000004, 62.03703703703704), (0.032701, 62.03703703703704), (0.034701, 62.03703703703704), (0.036501000000000006, 62.03703703703704), (0.037601, 62.03703703703704), (0.039501, 62.03703703703704), (0.043601, 62.03703703703704), (0.061701000000000006, 62.03703703703704), (0.063101, 62.03703703703704), (0.06330100000000001, 62.03703703703704), (0.064901, 62.03703703703704), (0.065501, 62.03703703703704), (0.071901, 62.03703703703704), (0.075201, 62.03703703703704), (0.089601, 62.03703703703704), (0.09000100000000001, 62.03703703703704)]

保守起见，我们令第二个值为最优复杂惩罚参数值，即0.0041

**最后，隐藏层节点个数调优：**

范围限制：【3，25】

```python
accu = []
for i in np.arange(3, 100, 1):
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
plt.plot(np.arange(3, 100, 1), accu)
plt.suptitle('Accuracy-hidden_Layer_NUM')
plt.xlabel('hidden_Layer_NUM')
plt.ylabel('Accuracy%')
plt.show()
```

**控制台部分输出结果：**

混淆矩阵如下：
 [[58  9]
 [35  6]]
预测精度为:59.26 %
混淆矩阵如下：
 [[46 21]
 [26 15]]
预测精度为:56.48 %
混淆矩阵如下：
 [[53 14]
 [29 12]]
预测精度为:60.19 %
混淆矩阵如下：
 [[53 14]
 [30 11]]
预测精度为:59.26 %
混淆矩阵如下：
 [[48 19]
 [26 15]]
预测精度为:58.33 %
混淆矩阵如下：
 [[52 15]
 [31 10]]
预测精度为:57.41 %
混淆矩阵如下：
 [[48 19]
 [30 11]]
预测精度为:54.63 %
混淆矩阵如下：
 [[49 18]
 [25 16]]
预测精度为:60.19 %
混淆矩阵如下：
 [[48 19]
 [27 14]]
预测精度为:57.41 %
混淆矩阵如下：
 [[53 14]
 [25 16]]
预测精度为:63.89 %
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.

混淆矩阵如下：
Increase the number of iterations (max_iter) or scale the data as shown in:
 [[50 17]
    https://scikit-learn.org/stable/modules/preprocessing.html
 [26 15]]
预测精度为:60.19 %
  self.n_iter_ = _check_optimize_result("lbfgs", opt_res, self.max_iter)
C:\Users\DELL\anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:470: ConvergenceWarning: lbfgs failed to converge (status=1):
STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.....................

**图表输出：**

![image-20210325190241056](C:\Users\DELL\PycharmProjects\skearn_prediction\README.assets\image-20210325190241056.png)

对于本数据集，模型复杂度不宜过高（防止过拟合），因此隐藏层应尽量选择选择参数较小的值作为最优值。同样，打印出准确率较高的前几个相对优秀的隐藏层数目如下：

[(87, 64.81481481481481), (93, 64.81481481481481), (96, 64.81481481481481), (97, 64.81481481481481), (98, 64.81481481481481), (99, 64.81481481481481), (105, 64.81481481481481), (106, 64.81481481481481), (113, 64.81481481481481), (117, 64.81481481481481), (118, 64.81481481481481), (119, 64.81481481481481), (120, 64.81481481481481), (121, 64.81481481481481), (122, 64.81481481481481), (123, 64.81481481481481), (124, 64.81481481481481), (125, 64.81481481481481), (126, 64.81481481481481), (128, 64.81481481481481), (131, 64.81481481481481), (133, 64.81481481481481), (134, 64.81481481481481), (135, 64.81481481481481), (138, 64.81481481481481), (139, 64.81481481481481), (141, 64.81481481481481), (142, 64.81481481481481), (144, 64.81481481481481), (145, 64.81481481481481), (146, 64.81481481481481), (147, 64.81481481481481), (149, 64.81481481481481), (12, 63.888888888888886), (14, 62.96296296296296), (112, 62.03703703703704)]

最终选择12作为最优参数

## 模型缺陷

1，Sklearn调包默认预测概率值大于0.5即为类别1，0.5或许对于该数据集而言并非恰当的概率分隔值，分隔值过大或者过小都会导致误判率提升。实际过程中，应当采用ROC曲线工具，选取恰当的概率分隔值进行分割，以提高模型的预测精准度；

2，关于最优参数的选择，仍然需要做进一步的实验进行验证。如将数据重新打乱（但是维持验证集与测试集分布大致相同），观察该轮最优参数选择得出的准确率与新数据得到的准确率是否大致相同。

## 交流

问题可联系2393946194@qq.com，欢迎互相交流与学习！





