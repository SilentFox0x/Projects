import argparse
import os
import numpy as np
import math
import sys
sys.path.append('..')

import tensorflow as tf
import tensorflow_datasets as tfds
from models import *
from loader import *  # assumes loader.init_dataloader_tf or similar
from PIL import Image

device = "cuda:0" if tf.config.list_physical_devices('GPU') else "cpu"
gender_idx = 20
target_idx = gender_idx


def gradient_penalty(x, y):
    # interpolation
    shape = tf.shape(x)
    alpha = tf.random.uniform(shape=[shape[0]] + [1] * (len(shape) - 1), dtype=tf.float32)
    z = x + alpha * (y - x)
    z = tf.Variable(z)
    with tf.GradientTape() as tape:
        tape.watch(z)
        o = discriminator(z, training=True)
    g = tape.gradient(o, z)
    g_flat = tf.reshape(g, [shape[0], -1])
    gp = tf.reduce_mean((tf.norm(g_flat, axis=1) - 1.0) ** 2)
    return gp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training [deprecated]")
    parser.add_argument("--n_iters", type=int, default=100000, help="number of iterations of training")
    parser.add_argument("--latent_dim", type=int, default=128, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
    parser.add_argument("--data_num", type=int, default=4096, help="size of training dataset")
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--lr", type=float, default=4e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    opt = parser.parse_args()
    print(opt)

    latent_dim = opt.latent_dim
    bs = 64

    # Loss weight for gradient penalty
    lambda_gp = 10
    print("latent dimension ", latent_dim)
    print("learning rate", opt.lr)
    # Initialize generator and discriminator
    generator = GanGenerator(in_dim=latent_dim)
    generator.load_weights(f'../Gan/zdim_{latent_dim}/params/G_initial.h5')
    discriminator = GanDiscriminator(in_dim=2)
    discriminator.load_weights(f'../Gan/zdim_{latent_dim}/params/D_initial.h5')

    # Transforms
    def trans_n(x):
        mean = tf.constant([0.485, 0.456, 0.406])
        std = tf.constant([0.229, 0.224, 0.225])
        return (x - mean) / std

    def trans_crop(x):
        return tf.image.central_crop(x, central_fraction=128/Math.cast(tf.shape(x)[1], tf.float32))

    def trnas_resize(x):
        return tf.image.resize(x, [64, 64])

    def trans_tensor(x):
        return x

    def trans(x):
        return trans_tensor(trnas_resize(trans_crop(trans_n(x))))

    # Dataset
    ds = tfds.load('celeb_a', split='train', as_supervised=False)
    ds = ds.map(lambda d: tf.cast(d['image'], tf.float32) / 255.0)
    ds = ds.map(lambda img: trans(img))
    ds = ds.take(opt.data_num)
    dataloader = ds.batch(bs).prefetch(tf.data.AUTOTUNE)

    result_dir = f'../Gan/zdim_{opt.latent_dim}/images/'
    os.makedirs(result_dir, exist_ok=True)
    store_param = f'../Gan/zdim_{opt.latent_dim}/params/'
    os.makedirs(store_param, exist_ok=True)

    # Optimizers
    optimizer_G = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)
    optimizer_D = tf.keras.optimizers.Adam(learning_rate=opt.lr, beta_1=opt.b1, beta_2=opt.b2)

    # ----------
    #  Training
    # ----------
    batches_done = 0
    epoch = 0
    while batches_done < opt.n_iters:
        for i, batch in enumerate(dataloader):
            epoch += 1
            real_imgs = batch

            # Sample noise as generator input
            z = tf.random.normal((tf.shape(real_imgs)[0], latent_dim))

            # Generate a batch of images
            fake_imgs = generator(z, training=True)

            # Real images
            with tf.GradientTape() as tape_D:
                real_validity = discriminator(real_imgs, training=True)
                fake_validity = discriminator(fake_imgs, training=True)
                gp = gradient_penalty(real_imgs, fake_imgs)
                d_loss = -tf.reduce_mean(real_validity) + tf.reduce_mean(fake_validity) + lambda_gp * gp
            grads_D = tape_D.gradient(d_loss, discriminator.trainable_variables)
            optimizer_D.apply_gradients(zip(grads_D, discriminator.trainable_variables))

            # Train the generator every n_critic steps
            if (batches_done + 1) % opt.n_critic == 0:
                with tf.GradientTape() as tape_G:
                    fake_imgs = generator(z, training=True)
                    fake_validity = discriminator(fake_imgs, training=False)
                    g_loss = -tf.reduce_mean(fake_validity)
                grads_G = tape_G.gradient(g_loss, generator.trainable_variables)
                optimizer_G.apply_gradients(zip(grads_G, generator.trainable_variables))

                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                      % (epoch, opt.n_epochs, i, len(dataloader), d_loss.numpy(), g_loss.numpy()))
                print("gp is {}".format(gp.numpy()))

            batches_done += 1

            if batches_done % (opt.n_critic * 1000) == 0:
                generator.save_weights(store_param + f'G_{batches_done}.h5')
                discriminator.save_weights(store_param + f'D_{batches_done}.h5')
                # save_image equivalent
                imgs_np = (fake_imgs.numpy() * 255).astype(np.uint8)
                # tile and save...
    # Final save
    generator.save_weights(store_param + f'G_{batches_done}.h5')
    discriminator.save_weights(store_param + f'D_{batches_done}.h5')