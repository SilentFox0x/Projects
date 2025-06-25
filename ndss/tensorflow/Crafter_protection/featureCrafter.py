import sys
sys.path.append('..')
from WhiteBoxAttack.attacker import amor_inversion, Amortizer
from loader import init_dataloader, save_tensor_images, plot_loss
from models import Encoder, Decoder, GanDiscriminator, newGanGenerator
import os
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from typing import Tuple


def f_loss(f, f_rec):
    n = tf.shape(f)[0]
    d = 64 * 16 * 16
    f_flat = tf.reshape(f, [n, -1])
    f_rec_flat = tf.reshape(f_rec, [n, -1])
    mse = tf.reduce_sum(tf.square(f_flat - f_rec_flat), axis=1)
    mse = mse / tf.cast(d, mse.dtype)
    return tf.reduce_mean(mse)


def solve_z(feature, G, Enc):
    # Find initial x* of feature
    tf.keras.backend.set_learning_phase(0)
    # Amortized inversion
    z = amor_inversion(G, D, Enc, Amor, feature, z_dim=zd)
    z = tf.identity(z)
    z = tf.Variable(z, trainable=True)
    return z


def gradient_penalty(x, y):
    batch_size = tf.shape(x)[0]
    shape = [batch_size] + [1] * (len(x.shape) - 1)
    alpha = tf.random.uniform(shape)
    interpolate = x + alpha * (y - x)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interpolate)
        o = netD(interpolate)
    grads = gp_tape.gradient(o, interpolate)
    grads = tf.reshape(grads, [batch_size, -1])
    gp = tf.reduce_mean((tf.norm(grads, axis=1) - 1.0) ** 2)
    return gp


def gradient_based_ho(f0, netD, batch_id):
    f_adv = tf.Variable(f0, trainable=True)
    f_optimizer = tf.keras.optimizers.Adam(learning_rate=flr)
    d_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4)
    p_list, u_list = [], []
    z_avg = tf.random.normal([bs, zd], dtype=tf.float32)

    for epoch in range(epochs):
        print(f'\nepoch {epoch}')
        z = solve_z(f_adv, G, Enc)

        # Update Discriminator
        for _ in range(10):
            real_index = tf.random.uniform([d_batch], maxval=bs, dtype=tf.int32)
            fake_index = tf.random.uniform([d_batch], maxval=bs, dtype=tf.int32)
            real = G(z_avg)[real_index]
            fake = G(z)[fake_index]

            with tf.GradientTape() as d_tape:
                neg_logit = netD(fake)
                pos_logit = netD(real)
                gp = gradient_penalty(real, fake)
                l_gp = tf.where(gp > 1.0, 40.0, lambda_gp)
                EMD = tf.reduce_mean(neg_logit) - tf.reduce_mean(pos_logit)
                d_loss = EMD + l_gp * gp
            grads = d_tape.gradient(d_loss, netD.trainable_variables)
            d_optimizer.apply_gradients(zip(grads, netD.trainable_variables))

            p_list.append(EMD.numpy())
            print(f" gp {gp.numpy()}, d loss {d_loss.numpy()} ")

        # Inversion and hypergradient update
        with tf.GradientTape() as tot_tape, tf.GradientTape() as inv_tape:
            inv_out = Enc(G(z))
            inv_loss = f_loss(f_adv, inv_out) * 2000.0
            tot = 100 * -tf.reduce_mean(netD(G(z))) + 100 * beta * f_loss(f_adv, f0)
        tot_grads = tot_tape.gradient(tot, [f_adv])
        inv_grads_z = inv_tape.gradient(inv_loss, [z])

        hyper_grads = hypergradient(tot, inv_loss, f_adv, z)
        f_adv_grad = hyper_grads[0]
        print(f'check feature grad is {tf.norm(f_adv_grad)}')
        f_optimizer.apply_gradients([(f_adv_grad, f_adv)])

        u_list.append(f_loss(f_adv, f0).numpy())
        if epoch % 3 == 0:
            save_tensor_images(G(z).numpy(), f'{result_dir}/images/inv_{epoch}.jpg')
            tf.io.write_file(f'{result_dir}/features/{batch_id}_{epoch}.pt', tf.io.serialize_tensor(f_adv))

    plot_loss(p_list, result_dir + '/curves', f'privacy_{batch_id}', 'EM distance')
    plot_loss(u_list, result_dir + '/curves', f'utility_{batch_id}', 'feature norm')


def hypergradient(tot_loss, inv_loss, f_adv, z):
    # Compute gradients
    v1 = tf.gradients(tot_loss, z)
    d_inv_d_z = tf.gradients(inv_loss, z)
    v2 = approxInverseHVP(v1, d_inv_d_z, z, i=i, alpha=alpha)
    v3 = tf.gradients(d_inv_d_z, f_adv, grad_ys=v2)
    d_tot_d_f = tf.gradients(tot_loss, f_adv)
    return [d - v for d, v in zip(d_tot_d_f, v3)]


def approxInverseHVP(v, f, z, i=30, alpha=.01):
    p = v
    for j in range(i):
        grad_f = tf.gradients(f, z, grad_ys=v)
        if j % 30 == 0:
            print(f'inner epoch {j}, grad is {tf.norm(grad_f[0])}, v is {tf.norm(v[0])}')
        v = [v_ - alpha * g for v_, g in zip(v, grad_f)]
        p = [p_ + v_ for p_, v_ in zip(p, v)]
    p = [alpha * p_ for p_ in p]
    return p

if __name__ == "__main__":
    device = "gpu"
    bs = 128
    epochs = 450
    zd = 500
    lambda_gp = 20.0
    beta_list = [10]
    flr, alpha = .01, .001
    i_list = [120]
    d_batch = 32

    _, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1)

    G = newGanGenerator(in_dim=zd)
    G.load_weights(f'../Gan/zdim_{zd}/params/G_90.h5')
    G.trainable = False

    D = GanDiscriminator()
    D.load_weights(f'../Gan/zdim_{zd}/params/D_90.h5')
    D.trainable = False

    Enc = Encoder()
    Enc.model.fc = tf.keras.layers.Dense(40)
    Enc.load_weights('../params/enc.h5')
    Enc.trainable = False

    Amor = Amortizer(nz=zd)
    Amor.load_weights('../params/amor.h5')
    Amor.trainable = False

    Dec = Decoder()
    Dec.load_weights('../params/dec.h5')
    Dec.trainable = False

    for beta in beta_list:
        for i in i_list:
            result_dir = f'../Crafter_result/beta{beta}/'
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(result_dir + '/original', exist_ok=True)
            os.makedirs(result_dir + '/images', exist_ok=True)
            os.makedirs(result_dir + '/features', exist_ok=True)
            os.makedirs(result_dir + '/curves', exist_ok=True)
            for batch_id, (imgs, label) in enumerate(dataloader):
                tf.keras.backend.clear_session()
                imgs = tf.convert_to_tensor(imgs)
                save_tensor_images(imgs.numpy(), f'{result_dir}/original/{batch_id}.jpg')
                f0 = Enc(imgs).numpy()
                netD = GanDiscriminator()
                netD.load_weights('../Gan/zdim_{zd}/params/D_90.h5')
                netD.trainable = True
                gradient_based_ho(f0, netD, batch_id)
