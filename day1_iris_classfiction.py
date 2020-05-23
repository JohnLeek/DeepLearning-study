import tensorflow as tf
from sklearn import datasets
from matplotlib import pyplot as plt
import numpy as np

#导入数据
x_data = datasets.load_iris().data
y_data = datasets.load_iris().target

'''
打乱顺序
seed为随机种子，保证每次生成的随机数一样
'''
np.random.seed(116)
np.random.shuffle(x_data)
np.random.seed(116)
np.random.shuffle(y_data)
tf.random.set_seed(116)

#拆分训练集和测试集，前120组作为训练集，后30组作为测试机
x_train = x_data[:-30]
y_train = y_data[:-30]

x_test = x_data[-30:]
y_test = y_data[-30:]

#转化数据,将x都转化为同一类数据
x_train = tf.cast(x_train,tf.float32)
x_test = tf.cast(x_test,tf.float32)

'''
匹配输入特征还有标签
batch为喂入神经神经网络每组数据大小
'''
train_db = tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(32)
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test)).batch(32)

'''
搭建神经网络，4个特征值，输入层为4个神经元；3分类，所以输出层为3个神经元
seed保证生成的随机数相同，实际项目不需要
Variable标记训练参数
'''
w1 = tf.Variable(tf.random.truncated_normal([4,3],stddev=0.1,seed=1))
b1 = tf.Variable(tf.random.truncated_normal([3],stddev=0.1,seed=1))

lr = 0.1#学习率
train_loss_result = []#记录每轮训练之后的loss
test_acc = []#模型准确度
epoch = 500#每轮训练次数
loss_all = 0#每轮分为4step，记录四个step生成的4个loss和

#训练部分
for epoch in range(epoch): #数据集别循环每个epoch遍历一次数据集
    for step,(x_train,y_train) in enumerate(train_db): #batch级别循环，每个step循环一个batch
        with tf.GradientTape() as tape: #梯度下降求得参数值
            y = tf.matmul(x_train,w1) + b1 #函数表达式 y = xw + b
            y = tf.nn.softmax(y)#使输出符合概率分布
            y_ = tf.one_hot(y_train,depth = 3)#将标签转化为独热码，方便求loss和accuracy
            loss = tf.reduce_mean(tf.square(y_-y))#采用均方误差损失函数mes = mean(sum(y-out)^2)
            loss_all += loss.numpy()#将每个step计算得出的loss累加，为后续求loss平均值提供数据
        #计算w1和b1梯度
        grads = tape.gradient(loss,[w1,b1])
        #更新梯度
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr*grads[1])
    #每个epoch打印下loss
    print("Epoch: {},loss: {}".format(epoch,loss_all/4))
    #保存每个step的平均值
    train_loss_result.append(loss_all/4)
    loss_all = 0 #归零，为下一次循环做准备

    #测试部分
    '''
    total_correct为预测正确的样本个数
    total_number为测试样本总数
    '''
    total_correct,total_number = 0,0
    for x_test,y_train in test_db:
        y = tf.matmul(x_test, w1) + b1
        y = tf.nn.softmax(y)
        pred = tf.argmax(y, axis=1)  # 返回y中的最大值索引，即预测分类
        # 将pred转化为y_test数据类型
        pred = tf.cast(pred, dtype=y_test.dtype)
        # 如果分类正确，correct = 1 否则为0,将bool类型结果转化为int
        correct = tf.cast(tf.equal(pred, y_test), dtype=tf.int32)
        # 将每个batch的correct数加起来
        correct = tf.reduce_sum(correct)
        # 将所有batch中的correct加起来
        total_correct += int(correct)
        # total_number为测试的总样本数，即x_test的行数,shape[0]返回行数
        total_number += x_test.shape[0]
    # 计算准确率
    acc = total_correct / total_number
    test_acc.append(acc)
    print("Test_acc:", acc)
    print("这是一条分割线------------------********************++++++++++++++++++")

#绘制loss曲线
plt.title("Losss Function Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.plot(train_loss_result,label="$Loss$")# 逐点画出trian_loss_results值并连线，连线图标是Loss
plt.legend()
plt.savefig("./loss")

#绘制Accuracy曲线
plt.title("Acc Curve")
plt.xlabel("Epoch")
plt.ylabel("Acc")
plt.plot(test_acc,label="$Accuracy$") # 逐点画出test_acc值并连线，连线图标是Accuracy
plt.legend
plt.savefig("./acc")




































































