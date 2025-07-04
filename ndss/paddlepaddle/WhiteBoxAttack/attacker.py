import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import os
import random
import sys

import paddle
from paddle_utils import *

sys.path.append("..")
import argparse
import csv
from statistics import mean

import matplotlib.pyplot as plt
import numpy
import sklearn.metrics
from facenet_pytorch import InceptionResnetV1
from loader import *
from metrics import *
from models import *


def my_auc(labelll, preddd):
    auclist = []
    labelll = labelll.detach().cpu()
    preddd = preddd.detach().cpu()
    for i in range(tuple(label.shape)[1]):
        try:
            a = sklearn.metrics.roc_auc_score(labelll[:, i], preddd[:, i])
        except:
            pass
        else:
            auclist.append(a)
    return auclist


def plot_loss(loss_list, directory, name, title):
    """
    name: filename
    title: plot title
    """
    x = range(len(loss_list))
    plt.plot(x, loss_list)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Distance")
    if not os.path.exists(directory):
        os.makedirs(directory)
    plt.savefig(directory + "/" + name + ".png")
    plt.close()


def inversion(
    G,
    Enc,
    feature_real,
    result_dir,
    name="",
    title="",
    lr=0.005,
    momentum=0.9,
    iter_times=601,
    z_dim=500,
    save=True,
):
    """
    White box inversion, momentum descent
    G: generator
    Enc: target network
    feature_real: feature to invert
    result_dir: where to store inverted images
    batch: to generate random seed
    name: inversion loss curve file name
    tilte: inversion loss curve title
    """
    bs = tuple(feature_real.shape)[0]
    G.eval()
    Enc.eval()
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    z = paddle.randn(shape=[bs, z_dim]).cuda().astype(dtype="float32")
    z.stop_gradient = not True
    v = paddle.zeros(shape=[bs, z_dim]).cuda().astype(dtype="float32")
    loss_list = []
    for _ in range(iter_times):
        fake = G(z)
        feature_fake = Enc(fake)
        if z.grad is not None:
            z.grad.data.zero_()
        feature_loss = paddle.dist(x=feature_fake, y=feature_real, p=2)
        total_loss = 100 * feature_loss
        total_loss.backward()
        loss_list.append(feature_loss.data.float().cpu())
        v_prev = v.clone()
        gradient = z.grad.data.clone()
        v = momentum * v_prev + gradient
        z = z - lr * v
        z = z.detach()
        z.stop_gradient = not True
    if save:
        save_tensor_images(fake.detach(), result_dir + "/inverted.jpg", nrow=8)
        plot_loss(loss_list, result_dir, "loss curve", "inv loss")
    return z


def amor_inversion(
    G,
    D,
    Enc,
    Amor,
    feature_real,
    result_dir="",
    lr=0.005,
    momentum=0.9,
    iter_times=80,
    z_dim=500,
    save=False,
):
    """
    White box inversion, using amor to set initial z
    G: generator
    Enc: target network
    feature_real: feature to invert
    """
    G.eval()
    Amor.eval()
    Enc.eval()
    bs = tuple(feature_real.shape)[0]
    z = Amor(feature_real).clone().detach().cuda()
    z.stop_gradient = not True
    v = paddle.zeros(shape=[bs, z_dim]).cuda().astype(dtype="float32")
    loss_list = []
    for _ in range(iter_times):
        fake = G(z)
        feature_fake = Enc(fake)
        if z.grad is not None:
            z.grad.data.zero_()
        feature_loss = paddle.dist(x=feature_fake, y=feature_real, p=2)
        prior_loss = -D(fake).mean()
        total_loss = 100 * feature_loss + 100 * prior_loss
        total_loss.backward()
        loss_list.append(feature_loss.data.float().cpu())
        v_prev = v.clone()
        gradient = z.grad.data.clone()
        v = momentum * v_prev + gradient
        z = z - lr * v
        z = z.detach()
        z.stop_gradient = not True
    if save:
        save_tensor_images(fake.detach(), result_dir + "/inverted.jpg", nrow=8)
        plot_loss(loss_list, result_dir, "loss curve", "inv loss")
    return z
