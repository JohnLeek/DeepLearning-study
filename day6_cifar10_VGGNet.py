# noinspection PyUnresolvedReferences
import tensorflow as tf
import numpy as np
import os
# noinspection PyUnresolvedReferences
from tensorflow.keras.layers import Conv2D,BatchNormalization,Activation,MaxPool2D,Dropout,Flatten,Dense
# noinspection PyUnresolvedReferences
from tensorflow.keras import Model
from matplotlib import pyplot as plt

'''
解决cudnn无法加载，动态分配显卡内存
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
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
x_train,x_test = x_train/225.0,x_test/255.0

class VGGNet(Model):
    def __init__(self):
        super(VGGNet,self).__init__()
        self.c1 = Conv2D(filters = 64,kernel_size = (3,3),padding = "same")
        self.b1 = BatchNormalization()
        self.a1 = Activation("relu")

        self.c2 = Conv2D(filters = 64,kernel_size = (3,3),padding = "same")
        self.b2 = BatchNormalization()
        self.a2 = Activation("relu")
        self.p2 = MaxPool2D(pool_size = (2,2),strides = 2)
        self.d2 = Dropout(0.2)

        self.c3 = Conv2D(filters = 128,kernel_size = (3,3),padding = "same")
        self.b3 = BatchNormalization()
        self.a3 = Activation("relu")

        self.c4 = Conv2D(filters = 128,kernel_size = (3,3),padding = "same")
        self.b4 = BatchNormalization()
        self.a4 = Activation("relu")
        self.p4 = MaxPool2D(pool_size = (2,2),strides = 2)
        self.d4 = Dropout(0.2)

        self.c5 = Conv2D(filters = 256,kernel_size = (3,3),padding = "same")
        self.b5 = BatchNormalization()
        self.a5 = Activation("relu")

        self.c6 = Conv2D(filters=256, kernel_size=(3, 3), padding="same")
        self.b6 = BatchNormalization()
        self.a6 = Activation("relu")

        self.c7 = Conv2D(filters=128, kernel_size=(3, 3), padding="same")
        self.b7 = BatchNormalization()
        self.a7 = Activation("relu")
        self.p7 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d7 = Dropout(0.2)

        self.c8 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.b8 = BatchNormalization()
        self.a8 = Activation("relu")

        self.c9 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.b9 = BatchNormalization()
        self.a9 = Activation("relu")

        self.c10 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.b10 = BatchNormalization()
        self.a10 = Activation("relu")
        self.p10 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d10 = Dropout(0.2)

        self.c11 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.b11 = BatchNormalization()
        self.a11 = Activation("relu")

        self.c12 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.b12 = BatchNormalization()
        self.a12 = Activation("relu")

        self.c13 = Conv2D(filters=512, kernel_size=(3, 3), padding="same")
        self.b13 = BatchNormalization()
        self.a13 = Activation("relu")
        self.p13 = MaxPool2D(pool_size=(2, 2), strides=2)
        self.d13 = Dropout(0.2)

        self.flatten = Flatten()
        self.f14 = Dense(512,activation = "relu")
        self.d14 = Dropout(0.2)
        self.f15 = Dense(512,activation = "relu")
        self.d15 = Dropout(0.2)
        self.f16 = Dense(10,activation = "softmax")

    def call(self,x):
        x = self.c1(x)
        x = self.b1(x)
        x = self.a1(x)

        x = self.c2(x)
        x = self.b2(x)
        x = self.a2(x)
        x = self.p2(x)
        x = self.d2(x)

        x = self.c3(x)
        x = self.b3(x)
        x = self.a3(x)

        x = self.c4(x)
        x = self.b4(x)
        x = self.a4(x)
        x = self.p4(x)
        x = self.d4(x)

        x = self.c5(x)
        x = self.b5(x)
        x = self.a5(x)

        x = self.c6(x)
        x = self.b6(x)
        x = self.a6(x)

        x = self.c7(x)
        x = self.b7(x)
        x = self.a7(x)
        x = self.p7(x)
        x = self.d7(x)

        x = self.c8(x)
        x = self.b8(x)
        x = self.a8(x)

        x = self.c9(x)
        x = self.b9(x)
        x = self.a9(x)

        x = self.c10(x)
        x = self.b10(x)
        x = self.a10(x)
        x = self.p10(x)
        x = self.d10(x)

        x = self.c11(x)
        x = self.b11(x)
        x = self.a11(x)

        x = self.c12(x)
        x = self.b12(x)
        x = self.a12(x)

        x = self.c13(x)
        x = self.b13(x)
        x = self.a13(x)
        x = self.p13(x)
        x = self.d13(x)

        x = self.flatten(x)
        x = self.f14(x)
        x = self.d14(x)
        x = self.f15(x)
        x = self.d15(x)
        y = self.f16(x)

        return y

model = VGGNet()

model.compile(
    optimizer = "adam",
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = False),
    metrics = ["sparse_categorical_accuracy"]
)

#断点续训
checkpoint_save_path = "./checkpoint_VGGNet/VGGNet.ckpt"
if os.path.exists(checkpoint_save_path):
    print("-------------------Load VGGNet Model---------------------")
    model.load_weights(checkpoint_save_path)

cp_callbacks = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_save_path,
    save_weights_only = True,
    save_best_only = True
)

history = model.fit(
    x_train,y_train,batch_size = 128,epochs = 25,validation_data = (x_test,y_test),
    validation_freq = 1,callbacks = [cp_callbacks]
)

model.summary()

file = open("./cifar10_VGGNet.txt","w")
for v in model.trainable_variables:
    file.write(str(v.name)+"\n")
    file.write(str(v.shape)+"\n")
    file.write(str(v.numpy())+"\n")
file.close()

#取acc  loss
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
plt.title("Training and Validation Loss")
plt.legend()

plt.savefig("./VGGNet_two")