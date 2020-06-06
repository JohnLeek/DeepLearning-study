# noinspection PyUnresolvedReferences
import tensorflow as tf
import os
import numpy as np
from matplotlib import pyplot as plt
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
# noinspection PyUnresolvedReferences
from tensorflow.keras import Model
'''
解决cudnn无法加载，动态分配内存
'''
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import ConfigProto
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


np.set_printoptions(threshold=np.inf)

cifar10 = tf.keras.datasets.cifar10
(x_train,y_trian),(x_test,y_test) = cifar10.load_data()
x_train,x_test = x_train/255.0,x_test/255.0

class Baseline(Model):
    def __init__(self):
        super(Baseline,self).__init__()
        self.c1 = Conv2D(filters=6,kernel_size=(5,5),padding="same")#卷积层
        self.b1 = BatchNormalization()#BN层
        self.a1 = Activation("relu")
        self.p1 = MaxPool2D(pool_size=(2,2),strides=2,padding="same")#池化层
        self.d1 = Dropout(0.2)

        self.c2 = Conv2D(filters=12,kernel_size=(5,5),padding="same")
        self.b2 = BatchNormalization()
        self.a2 = Activation("relu")

        self.c3 = Conv2D(filters=24,kernel_size=(5,5),padding="same")
        self.b3 = BatchNormalization()
        self.a3 = Activation("relu")
        self.p3 = MaxPool2D(pool_size=(2,2),strides=2,padding="same")
        self.d3 = Dropout(0.2)

        self.flatten = Flatten()
        self.f1 = Dense(128,activation="relu")
        self.d4 = Dropout(0.2)
        self.f2 = Dense(128,activation="relu")
        self.d5  =Dropout(0.2)
        self.f3 = Dense(10,activation="softmax")

    def call(self,x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)
        x = self.p1(x)
        x = self.d1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)
        x = self.p3(x)
        x = self.d3(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.d4(x)
        x = self.f2(x)
        x = self.d5(x)
        y = self.f3(x)
        return y

model = Baseline()

model.compile(optimizer = tf.keras.optimizers.Adam(
    learning_rate = 0.001,beta_1 = 0.9,beta_2 = 0.999#自定义动量，adam和学习率，也是为了提高准确率
),
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics = ["sparse_categorical_accuracy"]
              )
checkpoint_save = "./checkpoinCNNBase/Baseline.ckpt"
if os.path.exists(checkpoint_save+".index"):
    print("---------------load the model----------------------")
    model.load_weights(checkpoint_save)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_save,
    save_weights_only = True,
    save_best_only = True
)

history = model.fit(
    x_train,y_trian,batch_size = 2048,epochs = 300,validation_data = (x_test,y_test),
    validation_freq = 1,callbacks = [cp_callback]
)

model.summary()

file = open("./cifar10_weights.txt","w")
for v in model.trainable_variables:
    file.write(str(v.name)+"\n")
    file.write(str(v.shape)+"\n")
    file.write(str(v.numpy())+"\n")
file.close()

acc = history.history["sparse_categorical_accuracy"]
val_acc = history.history["val_sparse_categorical_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

plt.subplot(1,2,1)
plt.plot(acc,label = "Training Acc")
plt.plot(val_acc,label = "Validation Acc")
plt.title("Training and Validation Acc")
plt.legend()

plt.subplot(1,2,2)
plt.plot(loss,label = "Training Loss")
plt.plot(val_loss,label = "Validation Loss")
plt.title("Trainning and Validation Loss")
plt.legend()

plt.savefig("./cifar1_baseline")










