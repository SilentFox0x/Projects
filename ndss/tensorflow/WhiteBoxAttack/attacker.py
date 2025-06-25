import random
import sys
sys.path.append('..')
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from statistics import mean
import argparse
import csv

def my_auc(labelll, preddd):
    auclist = []
    labelll = labelll.numpy() if isinstance(labelll, tf.Tensor) else labelll
    preddd = preddd.numpy() if isinstance(preddd, tf.Tensor) else preddd
    for i in range(labelll.shape[1]):
        try:
            a = roc_auc_score(labelll[:, i], preddd[:, i])
        except:
            continue
        else:
            auclist.append(a)
    return auclist

def plot_loss(loss_list, directory, name, title):
    x = range(len(loss_list))
    plt.plot(x, loss_list)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Distance')
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(os.path.join(directory, name + '.png'))
    plt.close()

def inversion(G, Enc, feature_real, result_dir, name='', title='',
              lr=5e-3, momentum=0.9, iter_times=601, z_dim=500, save=True):
    bs = feature_real.shape[0]
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    z = tf.Variable(tf.random.normal([bs, z_dim]), trainable=True)
    v = tf.zeros_like(z)

    loss_list = []
    for _ in range(iter_times):
        with tf.GradientTape() as tape:
            fake = G(z, training=False)
            feature_fake = Enc(fake, training=False)
            feature_loss = tf.norm(feature_fake - feature_real, ord=2)
            total_loss = 100 * feature_loss

        grads = tape.gradient(total_loss, [z])[0]
        v = momentum * v + grads
        z.assign_sub(lr * v)
        loss_list.append(feature_loss.numpy())

    if save:
        save_tensor_images(fake, os.path.join(result_dir, "inverted.jpg"))
        plot_loss(loss_list, result_dir, 'loss curve', 'inv loss')
    return z

def amor_inversion(G, D, Enc, Amor, feature_real, result_dir='',
                   lr=5e-3, momentum=0.9, iter_times=80, z_dim=500, save=False):
    bs = feature_real.shape[0]

    z = Amor(feature_real, training=False)
    z = tf.Variable(z, trainable=True)
    v = tf.zeros_like(z)

    loss_list = []
    for _ in range(iter_times):
        with tf.GradientTape() as tape:
            fake = G(z, training=False)
            feature_fake = Enc(fake, training=False)
            feature_loss = tf.norm(feature_fake - feature_real, ord=2)
            prior_loss = - tf.reduce_mean(D(fake, training=False))
            total_loss = 100 * feature_loss + 100 * prior_loss

        grads = tape.gradient(total_loss, [z])[0]
        v = momentum * v + grads
        z.assign_sub(lr * v)
        loss_list.append(feature_loss.numpy())

    if save:
        save_tensor_images(fake, os.path.join(result_dir, "inverted.jpg"))
        plot_loss(loss_list, result_dir, 'loss curve', 'inv loss')
    return z

def save_tensor_images(images, filename, nrow=8):
    import math
    from PIL import Image

    images = images.numpy()
    images = np.clip(images * 255.0, 0, 255).astype(np.uint8)
    b, h, w, c = images.shape
    ncol = math.ceil(b / nrow)
    grid = Image.new('RGB', (w * nrow, h * ncol))
    for idx in range(b):
        row = idx // nrow
        col = idx % nrow
        img = Image.fromarray(images[idx])
        grid.paste(img, (col * w, row * h))
    grid.save(filename)