import random
import sys
import os
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, ops
import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import metrics as sklearn_metrics
from statistics import mean
sys.path.append('..')
from models import *
from loader import *

# 设置运行上下文
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

PRT = 'IFT'
bs = 128
max_epoch = 50
num_class = 40

for beta in [None]:
    mode = f'beta{beta}'

    figurePath = "../dataset/private"
    attributePath = "../dataset/attri_test.txt"
    result_dir = f"../params/cf/"
    os.makedirs(result_dir, exist_ok=True)
    print("Loading images from:", figurePath)
    print("TrainCF")
    print("Result Dir:", result_dir)

    _, dataloader = init_dataloader(attributePath, figurePath, action='prt', batch_size=bs, n_classes=num_class, skiprows=1, allAttri=True)

    Enc = Encoder()
    Enc.set_train(False)
    cf = Classifier(num_class)
    cf.set_train(True)

    opt = nn.Adam(cf.trainable_params(), learning_rate=0.001)
    loss_fn = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    loss_list = []

    for e in range(max_epoch):
        print("Epoch", e)
        # MindSpore DataLoader 迭代器
        for i, (img, label) in enumerate(dataloader.create_tuple_iterator()):
            # img, label 已为 MindSpore Tensor
            label = label.astype(ms.int32)
            # 特征提取不更新梯度
            feature = Enc(img)
            feature = ops.stop_gradient(feature)

            # 清空梯度
            opt.clear_grad()
            # 前向计算
            pred = cf(feature)
            # 计算损失
            loss = loss_fn(pred, label)
            loss_list.append(loss.asnumpy())
            # 反向传播并更新
            loss.backward()
            opt.step()

        # 保存模型参数
        ms.save_checkpoint(cf, os.path.join(result_dir, f"cf_{e}.ckpt"))
