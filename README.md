# Tensorflow-study
该部分为自己在学习tensorflow2.0中实现的各种模型还有算法，供大家参考  
day1为tenforlow入门。  
day2实现了正则化操作，dot.csv为day要使用的数据集。  
day3利用tensorflow中的keras模块搭建了神经网络实现了对mnist中手写数字的识别，然后引入了断点续训保存了训练好的模型。  
————day3_mnist_reg.py为keras搭建的基础模型  
————day3_mnist_trian_ex4 我引入了断点续训保存训练好的模型，并保存了神经网络参数权重  
day4使用tf搭建了一个简单的CNN  
————day4_cifar10_baseline.py为源码  
————day4_cifar1_baseline.png为模型结果  
day5使用tf实现了ResNet  
————day5_cifar10_ResNet.py为源码  
————day5_cifar10_ResNet.png为模型结果  
————day5_cifar10_ResNet_断点续训.png为断点续训得到模型结果，在这次断点续训之前我自己又跑了25次，所以模型结果比较高。  
day6使用tf实现了VGG16网络  
————day6_cifar10_VGG.py为源码  
————day6_cifar10_VGG.png为模型结果  
day7使用tf中的eager模式实现了花卉数据集的分类任务  
————day7_flowers_tf_data.py为源码  
————day7_flowers_tf_data.png为模型结果  
代码还在持续更新中，后边可能会在b站录讲解视频，后续会更新RNN和DCGAN  
