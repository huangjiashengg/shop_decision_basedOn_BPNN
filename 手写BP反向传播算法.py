# 1，通过数据识别输入层和输出层节点个数，并让用户指定隐藏层的节点个数---即确定神经网络的外貌
# 2，初始化权重和阈值
# 3，代入值计算第k个训练样本的损失函数，反向传播更新阈值和权重
# 4，重复第三个步骤，直到达到停止条件
# 5，重复第三第四步骤，直到遍历完所有的训练样本
# 此神经网络依赖于random包,math包,以及numpy包


import random
import math
import numpy as np
class neural_Network:
    def __init__(self, X, y, X_test=None, min_delta=0.00001, hidden_layers=(100, ), learning_rate=0.001, max_iter=300, alpha=0.03):
        # 输入训练数据集X和y,设置权重调整量阈值(小于该值则停止迭代)
        # hidden_layers输入隐藏层信息
        # learning_rate输入学习率
        # max_iter输入最大迭代次数
        # alpha输入模型复杂度参数
        self.training_datax = X
        self.training_datay = y
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_delta = min_delta
        self.alpha = alpha
        self.X_test = X_test

     # 获取输入节点的个数
    def inputNode_num(self):
        return len(self.training_datax[0])

    # 获取输出节点的个数
    def outputNode_num(self):
        return len(set(self.training_datay))

    # 对真实输出进行多值化(假设是多分类问题),这里默认输出将是0，1，2，3，...等中的一个
    # 表示（有几类便有几个输出），因此需要对进来的数值进行变换到上述值（0，1，2，3...）
    def multiValue_training_datay(self):
        training_datay_list = []
        for i in range(len(self.training_datay)):
            training_datay_li = []
            for j in range(len(set(self.training_datay))):
                training_datay_li.append(0)
            training_datay_li[int(self.training_datay[i])] = 1
            training_datay_list.append(training_datay_li)
        return training_datay_list


    # 获取隐藏层层数和相应层隐藏神经元个数,注意该函数输出两个值
    def hiddenLayers_num(self):
        floor = len(self.hidden_layers)
        hiddenLayers_num = {}
        if floor == 1:
            hiddenLayers_num[floor] = self.hidden_layers[0]
        else:
            for i in range(0, floor):
                hiddenLayers_num[i+1] = self.hidden_layers[i]
        return hiddenLayers_num

    def init_weight(self, weight_num):
        # random.seed(0) # 应当在调用此函数时使用
        x = np.arange(-1, 1, 0.001)
        x = tuple(x)
        return random.sample(x, weight_num)

    def produce_layweight(self):
    # 实实际际产生了一个嵌套随机权重列表
        if len(self.hiddenLayers_num()) == 1:
            weight_list = []
            for i in range(self.hiddenLayers_num()[1]):
                li = self.init_weight(self.inputNode_num())
                weight_list.append(li)
            return weight_list

    def produce_laythreshold(self):
        return self.init_weight(self.hiddenLayers_num()[1])

    # 输出层随机生成相应的权重和阈值
    def produce_outweight(self):
        weight_list = []
        for i in range(self.outputNode_num()):
            li = self.init_weight(self.hiddenLayers_num()[1])
            weight_list.append(li)
        return weight_list

    def produce_outthreshold(self):
        return self.init_weight(self.outputNode_num())

    # 定义单个神经元的加法器以及激活函数，得到输出
    def neuron(self, inputs, weight, threshold):
        for i in range(0, len(inputs)):
            sum = 0
            sum = sum + inputs[i]*weight[i]
            # inputs跟weights长度相同，但是跟threshold就不一定相同
            return self.function(sum-threshold)

    # 定义Sigmoid激活函数
    def function(self, value):
        return 1/(1+math.exp(-value))

    # 需要输入值
    def start_training(self):
        lay_output_list = []
        output_list = []
        layweight = self.produce_layweight()
        laythreshold = self.produce_laythreshold()
        outweight = self.produce_outweight()
        outthreshold = self.produce_outthreshold()
        for I in range(0, len(self.training_datax)):  # 外层循环遍历样本量
            repeat_var = 0
            if repeat_var <= self.max_iter:

                for j in range(self.hiddenLayers_num()[1]):
                    lay_output = self.neuron(inputs=self.training_datax[I],
                                             weight=layweight[j],
                                             threshold=laythreshold[j])
                    lay_output_list.append(lay_output)


                for k in range(self.outputNode_num()):
                    output = self.neuron(inputs=lay_output_list,
                                         weight=outweight[k],
                                         threshold=outthreshold[k]
                                         )

                    output_list.append(output)  # 至此，完成第一个训练样本的输出
                    # 接下来，反向调整两层权重和阈值
                    # 首先，反向调整输入的第一层
                g_func_list = []
                delta_outweight_li = []
                for l in range(self.outputNode_num()):
                    g_func_li = []
                    for i in range(self.hiddenLayers_num()[1]):

                        multiValue_training_datay = self.multiValue_training_datay()
                        g = self.G_fun(output_list[l], multiValue_training_datay[I][l])
                        g_func_li.append(g)
                        delta_outweight = self.learning_rate * g * lay_output_list[i]
                        delta_outweight_li.append(math.fabs(delta_outweight))
                        outweight[l][i] = outweight[l][i] + delta_outweight
                        outthreshold[l] = -self.learning_rate * g
                    g_func_list.append(g_func_li)
                # 调整第二层
                delta_layweight_li = []
                for l in range(self.hiddenLayers_num()[1]):

                    for i in range(self.inputNode_num()):
                        su = 0
                        for o in range(self.outputNode_num()):
                            s = outweight[o][l] * g_func_list[o][l]
                            su = su + s
                        e = lay_output_list[l] * (1 - lay_output_list[l]) * su
                        delta_layweight = self.learning_rate * e * self.training_datax[I][i]
                        delta_layweight_li.append(math.fabs(delta_layweight))
                        layweight[l][i] = layweight[l][i] + delta_layweight
                        laythreshold[l] = -self.learning_rate * e
                repeat_var = repeat_var + 1
                if max(delta_layweight_li) <= self.min_delta or max(delta_outweight_li) <= self.min_delta:
                    break
        return layweight, laythreshold, outweight, outthreshold
    # def get_layweight(self):
    #     return self.start_training()[0]
    # def get_laythreshold(self):
    #     return self.start_training()[1]
    # def get_outweight(self):
    #     return self.start_training()[2]
    # def outthreshold(self):
    #     return self.start_training()[3]
# 至此完成参数训练步骤
    def G_fun(self, estimate, real_val):
        return estimate*(1-estimate)*(real_val-estimate)
# 下面编写predict函数
    def predict(self):
        re_output_list = []
        simple_output_list = []
        lw, lt, ow, ot = self.start_training()
        for I in range(len(self.X_test)):
            lay_output_list = []
            output_list = []
            for j in range(self.hiddenLayers_num()[1]):

                lay_output = self.neuron(inputs=self.X_test[I],
                                         weight=lw[j],
                                         threshold=lt[j])
                lay_output_list.append(lay_output)

            for k in range(self.outputNode_num()):
                output = self.neuron(inputs=lay_output_list,
                                     weight=ow[k],
                                     threshold=ot[k]
                                     )
                output_list.append(output)
            if output_list[0] < output_list[1]:
                simple_out = 1
            else:
                simple_out = 0
            simple_output_list.append(simple_out)
            re_output_list.append(output_list)
        return re_output_list, simple_output_list




#################################运行一下#################################################

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


# 一比一随机不放回抽样
def sampling(len):
    train_index = random.sample(range(0, len), len // 2)  # 应用random.sample也可以在某数值范围内生成随机数
    test_index = []
    for i in range(0, 431):
        if i not in train_index:
            test_index.append(i)
    return train_index, test_index

train_index, test_index = sampling(431)

# 训练集与测试集手工划分
X_train = [X[x] for x in train_index]
y_train = [y[x] for x in train_index]
X_test = [X[x] for x in test_index]
y_test = [y[x] for x in test_index]

# print(X_train)
# print(y_train)
clf = neural_Network(X_train, y_train, X_test)
print(clf.predict()[1])

















