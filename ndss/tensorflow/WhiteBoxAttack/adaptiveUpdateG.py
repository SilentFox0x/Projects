import os
import time
import matplotlib.pyplot as plt
import pickle
import imageio
import itertools
import tensorflow as tf
import numpy as np
from tensorflow.keras import optimizers
from loader import *
from models import *
from attacker import inversion


def loss_func(inverted_img, img):
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    inverted_img = (inverted_img - mean) / std
    img = (img - mean) / std
    return tf.norm(inverted_img - img)


if __name__ == "__main__":
    bs = 128
    z_dim = 500
    attributePath = "../dataset/identity_pub_CelebA.txt"
    figurePath = "../dataset/public"

    G = newGanGenerator(in_dim=z_dim)
    G.load_weights('../params/G_90.h5')
    Amor = Amortizer(nz=z_dim)
    Amor.load_weights('../params/amor.h5')

    D = GanDiscriminator()
    D.load_weights('../params/D_90.h5')
    Enc = Encoder()
    Enc.model.fc = tf.keras.layers.Dense(40)
    Enc.model.load_weights('../params/enc.h5')

    beta = 50
    zlr = 1e-4

    result_dir = f'../params/adaptiveG/beta{beta}_zlr{zlr}/'
    os.makedirs(result_dir + 'z/', exist_ok=True)
    os.makedirs(result_dir + 'G/', exist_ok=True)
    os.makedirs(result_dir + 'img/', exist_ok=True)
    epoch = 50
    g_optimizer = optimizers.RMSprop(learning_rate=0.001)
    d_score = []
    final_loss_list = []

    _, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1)

    for e in range(epoch):
        loss_list = []
        print("epoch", e)
        z_avg = tf.random.normal((8, 500))
        save_tensor_images(G(z_avg), os.path.join(result_dir, 'img/img_epoch_{}.jpg'.format(e)))

        for i, (imgs, label) in enumerate(dataloader):
            if i == 50:
                break

            released_feature = tf.convert_to_tensor(np.load(f"./Crafter_result/features/{i}_447.npy"))

            current_inv_z, _ = inversion(G, D, Enc, released_feature, result_dir, lr=5e-3, z_dim=z_dim, save=False)

            if i == 0:
                fake_val = -tf.reduce_mean(D(G(z_avg)))
                real_val = tf.reduce_mean(D(imgs))
                print("d score is {}".format(fake_val + real_val))
                d_score.append(fake_val + real_val)

            with tf.GradientTape() as tape:
                generated_imgs = G(current_inv_z)
                loss = loss_func(generated_imgs, imgs)
                loss_list.append(loss)
            grads = tape.gradient(loss, G.trainable_variables)
            g_optimizer.apply_gradients(zip(grads, G.trainable_variables))

        print("loss is {}".format(tf.reduce_mean(loss_list)))
        final_loss_list.append(tf.reduce_mean(loss_list))
        G.save_weights(result_dir + 'G/G_epoch_{}.h5'.format(e))
        np.save(result_dir + 'score.npy', d_score)
        np.save(result_dir + 'loss.npy', final_loss_list)