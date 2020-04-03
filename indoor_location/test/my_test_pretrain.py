# -*- coding: utf-8 -*-
"""
使用VGG网络训练cifar-10,保存训练过程权重
@author: Administrator
"""
from sklearn.preprocessing import LabelBinarizer
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import SGD
from keras.datasets import cifar10

# import argparse

# =============================================================================
# # construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-w", "--weights", required=True,
#                 help="path to weights directory")
# args = vars(ap.parse_args())
# =============================================================================

# 导入数据
print("loading cifar-10 data ....")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

# 将label转化为vectors
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# 初始化模型和参数

print("compiling model .....")

opt = SGD(lr=0.01, decay=0.1 / 40, momentum=0.9, nesterov=True)
model = VGG16.build(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,
              metrics=["accuracy"])

# 创建一个权重文件保存文件夹logs
log_dir = "logs/"
# 记录所有训练过程，每隔一定步数记录最大值
tensorboard = TensorBoard(log_dir=log_dir)
checkpoint = ModelCheckpoint(log_dir + "best_weights.h5",
                             monitor="val_loss",
                             mode='min',
                             save_weights_only=True,
                             save_best_only=True,
                             verbose=1,
                             period=1)

callback_lists = [tensorboard, checkpoint]

print("training network...")
H = model.fit(trainX, trainY, validation_data=(testX, testY),
              batch_size=64, epochs=40, callbacks=callback_lists, verbose=2)