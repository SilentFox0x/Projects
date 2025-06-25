import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import itertools
import os
import pickle
import sys
import time

import imageio
import matplotlib.pyplot as plt
import paddle
from paddle_utils import *

sys.path.append("..")
from attacker import inversion
from loader import *
from models import *


def loss_func(inverted_img, img):
    trans = paddle.vision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    inverted_img, img = trans(inverted_img), trans(img)
    bce = paddle.dist(x=inverted_img, y=img)
    return bce


if __name__ == "__main__":
    bs = 128
    z_dim = 500
    attributePath = "../dataset/identity_pub_CelebA.txt"
    figurePath = "../dataset/public"
    G = newGanGenerator(in_dim=z_dim)
    G.set_state_dict(state_dict=paddle.load(path=str("../params/G_90.pkl")))
    G = paddle.DataParallel(layers=G).cuda()
    G.eval()
    Amor = Amortizer(nz=z_dim)
    Amor.set_state_dict(state_dict=paddle.load(path=str("../params/amor.pkl")))
    Amor = paddle.DataParallel(layers=Amor).cuda()
    Amor.eval()
    D = GanDiscriminator()
    D.set_state_dict(state_dict=paddle.load(path=str("../params/D_90.pkl")))
    D = paddle.DataParallel(layers=D).cuda()
    D.eval()
    Enc = Encoder()
    Enc.model.fc = paddle.nn.Linear(in_features=512, out_features=40)
    Enc.model.load_state_dict(paddle.load(path=str("../params/enc.pt")))
    Enc = paddle.DataParallel(layers=Enc).cuda()
    Enc.eval()
    D.eval()
    Enc.eval()
    beta = 50
    zlr = 0.0001
    result_dir = f"../params/adaptiveG/beta{beta}_zlr{zlr}/"
    os.makedirs(result_dir + "z/", exist_ok=True)
    os.makedirs(result_dir + "G/", exist_ok=True)
    os.makedirs(result_dir + "img/", exist_ok=True)
    epoch = 50
    g_optimizer = paddle.optimizer.RMSProp(
        parameters=G.parameters(),
        learning_rate=0.001,
        weight_decay=0.0,
        epsilon=1e-08,
        rho=0.99,
    )
    d_score = []
    final_loss_list = []
    _, dataloader = init_dataloader(
        attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1
    )
    for e in range(epoch):
        loss_list = []
        print("epoch", e)
        z_avg = paddle.randn(shape=(8, 500)).cuda()
        save_tensor_images(G(z_avg), result_dir + "img/img_epoch_{}.jpg".format(e))
        for i, (imgs, label) in enumerate(dataloader):
            if i == 50:
                break
            imgs = imgs.cuda()
            released_feature = paddle.load(
                path=str(f"./Crafter_result/features/{i}_447.pt")
            )
            current_inv_z, _ = inversion(
                G,
                D,
                Enc,
                released_feature,
                result_dir,
                lr=0.005,
                z_dim=z_dim,
                save=False,
            )
            if i == 0:
                fake_val, real_val = -paddle.mean(x=D(G(z_avg))), paddle.mean(x=D(imgs))
                print("d score is {}".format(fake_val + real_val))
                d_score.append(fake_val + real_val)
            g_optimizer.clear_gradients(set_to_zero=False)
            loss = loss_func(G(current_inv_z), imgs)
            loss_list.append(loss)
            loss.backward()
            g_optimizer.step()
        print("loss is {}".format(paddle.mean(x=paddle.to_tensor(data=loss_list))))
        final_loss_list.append(paddle.mean(x=paddle.to_tensor(data=loss_list)))
        paddle.save(
            obj=G.module.state_dict(), path=result_dir + "G/G_epoch_{}.pkl".format(e)
        )
        paddle.save(obj=d_score, path=result_dir + "score.pt")
        paddle.save(obj=final_loss_list, path=result_dir + "loss.pt")
