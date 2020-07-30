#Author By:lyq
#Create Time:2020/7/8 18:53
#解决cudnn无法加载，动态分配显卡内存
import glob
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,BatchNormalization,MaxPool2D,Dropout,Dense,Flatten
'''
解决cudnn无法加载，动态分配显卡内存
'''
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import ConfigProto
# noinspection PyUnresolvedReferences
from tensorflow.compat.v1 import InteractiveSession
import numpy as np
from matplotlib import pyplot as plt
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
path = './flowers/*/*.jpg'

all_image_path = glob.glob(path)
all_image_label = []


for p in all_image_path:
    if p.split('\\')[1] == 'daisy':
        all_image_label.append(0)
    if p.split('\\')[1] == 'dandelion':
        all_image_label.append(1)
    if p.split('\\')[1] == 'rose':
        all_image_label.append(2)
    if p.split('\\')[1] == 'sunflower':
        all_image_label.append(3)
    if p.split('\\')[1] == 'tulip':
        all_image_label.append(4)

np.random.seed(5000)
np.random.shuffle(all_image_path)
np.random.seed(5000)
np.random.shuffle(all_image_label)

image_count = len(all_image_path)
flag = int(len(all_image_path)*0.8)

train_image_path = all_image_path[:flag]
test_image_path = all_image_path[-(image_count-flag):]

train_image_label = all_image_label[:flag]
test_image_label = all_image_label[-(image_count-flag):]

def load_preprogress_image(image_path,label):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image,channels=3)
    image = tf.image.resize(image,[100,100])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.5)
    image = tf.image.random_contrast(image, 0, 1)
    image = tf.cast(image,tf.float32)
    image = image / 255.
    label = tf.reshape(label,[1])
    return image,label


Batch_Size = 32

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_image_dataset = tf.data.Dataset.from_tensor_slices((train_image_path,train_image_label))
train_image_dataset = train_image_dataset.map(load_preprogress_image,num_parallel_calls=AUTOTUNE)
train_image_dataset = train_image_dataset.shuffle(flag).batch(Batch_Size)
train_image_dataset = train_image_dataset.prefetch(AUTOTUNE)

test_image_dataset = tf.data.Dataset.from_tensor_slices((test_image_path,test_image_label))
test_image_dataset = test_image_dataset.map(load_preprogress_image,num_parallel_calls=AUTOTUNE)
test_image_dataset = test_image_dataset.shuffle(image_count-flag).batch(Batch_Size)
test_image_dataset = test_image_dataset.prefetch(AUTOTUNE)

model = Sequential([
    Conv2D(16,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPool2D(2,2),
    Dropout(0.2),

    Conv2D(32,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPool2D(2,2),
    Dropout(0.2),

    Conv2D(64, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Dropout(0.2),

    Conv2D(128,(3,3),padding="same",activation="relu"),
    BatchNormalization(),
    MaxPool2D(2,2),
    Dropout(0.2),

    Conv2D(256, (3, 3), padding="same", activation="relu"),
    BatchNormalization(),
    MaxPool2D(2, 2),
    Dropout(0.2),

    Flatten(),
    Dense(512,activation="relu"),
    Dropout(0.2),
    Dense(128, activation="relu"),
    Dropout(0.2),
    Dense(5,activation="softmax")
])


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean('train_loss')
train_accracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean('test_loss')
test_accracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images,labels):
    with tf.GradientTape() as tape:
       prediction = model(images)
       loss = loss_object(labels,prediction)
    gradients = tape.gradient(loss,model.trainable_variables)
    optimizer.apply_gradients(zip(gradients,model.trainable_variables))
    train_loss(loss)
    train_accracy(labels,prediction)

@tf.function
def test_step(images,lables):
    prediction = model(images)
    t_loss = loss_object(lables,prediction)

    test_loss(t_loss)
    test_accracy(lables,prediction)

def main():
    EPOCHS = 50
    # trainLoss = []
    # trainAcc = []
    # testLoss = []
    # testAcc = []
    # x_ = [p for p in range(EPOCHS)]
    for epoch in range(EPOCHS):
        # 下个循环开始时指标归零
        train_loss.reset_states()
        train_accracy.reset_states()
        test_loss.reset_states()
        test_accracy.reset_states()
        for images, labels in train_image_dataset:
            train_step(images, labels)
            # trainLoss.append(train_loss.result())
            # trainAcc.append(train_accracy.result())
        for test_images, test_labels in test_image_dataset:
            test_step(test_images, test_labels)
            # testLoss.append(test_loss.result())
            # testAcc.append(test_accracy.result())
        template = 'Epoch:{},Loss:{:.4f},Accuracy:{:.4f},Test Loss:{:.4f},Test Accuracy:{:.4f}'
        print(template.format(epoch + 1, train_loss.result(), train_accracy.result() * 100, test_loss.result(),
                              test_accracy.result() * 100))
'''
    plt.subplot(1,2,1)
    plt.plot(trainAcc,label ='train_acc')
    plt.plot(testAcc,label ='test_acc')
    plt.title('ACC')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(trainLoss, label='train_loss')
    plt.plot(testLoss, label='test_loss')
    plt.title('Loss')
    plt.legend()

    plt.savefig('./flowers_tf_data')
'''

if __name__ == '__main__':
    main()