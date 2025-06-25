import random
import sys
import os
import tensorflow as tf
import sys
sys.path.append('..')
from metrics import *
from models import *
from loader import *
import matplotlib.pyplot as plt
import numpy
import argparse
from facenet_pytorch import InceptionResnetV1
import sklearn.metrics
from statistics import mean
import csv
import matplotlib.pyplot as plt


PRT='IFT'
bs = 128
max_epoch = 50
num_class = 40
for beta in [None]:
    mode=f'beta{beta}'


    figurePath = "../dataset/private"
    attributePath = "../dataset/attri_test.txt"
    result_dir = f"../params/cf/"
    os.makedirs(result_dir, exist_ok=True)
    print("Loading images from:", figurePath)
    print("TrainCF")
    print("Result Dir:", result_dir)

    _, dataloader = init_dataloader(attributePath, figurePath, action='prt', batch_size=bs, n_classes=num_class, skiprows=1, allAttri=True)


    Enc = Encoder()
    # DataParallel and cuda are no-ops in TF
    Enc.eval = lambda : None
    Enc.eval()

    cf = Classifier(num_class)
    cf.trainable = True
    cf.train = lambda : None
    cf.train()

    opt = tf.keras.optimizers.Adam()


    # loss_fn = torch.nn.BCEWithLogitsLoss()
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    loss_list = []
    for e in range(max_epoch):
        print("Epoch", e)
        for i, (img, label) in enumerate(dataloader):
            # img, label are tf.Tensors
            # No torch.no_grad needed
            feature = Enc(img, training=False)
            with tf.GradientTape() as tape:
                pred = cf(feature, training=True)
                loss = loss_fn(label, pred)
            grads = tape.gradient(loss, cf.trainable_variables)
            opt.apply_gradients(zip(grads, cf.trainable_variables))
            loss_list.append(loss.numpy())
        # plt.plot(loss_list)
        # plt.savefig(f"./trainCF_figs/trainCFloss_{PRT}_{mode}.png")
        # Saving model weights regularly
        cf.save_weights(os.path.join(result_dir, f"cf_{e}.ckpt"))