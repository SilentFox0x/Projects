import sys
sys.path.append('..')
from WhiteBoxAttack.attacker import *
from loader import *
from models import *
import os
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops, context
from tqdm import tqdm
from typing import Tuple

# 设置 MindSpore 上下文
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


def f_loss(f, f_rec):
    n, d = f.shape[0], 64 * 16 * 16
    diff = ops.Reshape()(f, (n, -1)) - ops.Reshape()(f_rec, (n, -1))
    mse = ops.ReduceSum()(diff ** 2, 1) / d
    return ops.ReduceMean()(mse)


def solve_z(feature, G, Enc):
    """
    Find initial x* of feature
    """
    G.zero_grad()
    Enc.zero_grad()

    print("with amortize")
    z = amor_inversion(G, D, Enc, Amor, feature, z_dim=zd)
    z = z.clone().detach().requires_grad_(True)
    G.zero_grad()
    Enc.zero_grad()
    return z


def gradient_penalty(x, y):
    # interpolation
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    alpha = mnp.random.rand(*shape).astype(ms.float32)
    interpolate = x + alpha * (y - x)
    interpolate.requires_grad = True

    o = netD(interpolate)
    # TODO: 使用 MindSpore GradOperation 实现梯度计算
    grad_op = ops.GradOperation(get_all=True, sens_param=True)
    ones = mnp.ones(o.shape, ms.float32)
    g_tuple = grad_op(netD, None)(interpolate, ones)
    g = ops.Reshape()(g_tuple[0], (interpolate.shape[0], -1))
    gp = ops.ReduceMean()((ops.Norm()(g, p=2, axis=1) - 1) ** 2)

    return gp


def gradient_based_ho(f0, netD, batch_id):
    f_adv = f0.clone().detach().requires_grad_(True)
    f_optimizer = nn.Adam([f_adv], learning_rate=flr)
    d_optimizer = nn.Adam(netD.get_parameters(), learning_rate=5e-4)
    p_list, u_list = [], []
    z_avg = mnp.random.randn(bs, zd).astype(ms.float32)

    for epoch in range(epochs):
        print(f'\nepoch {epoch}')
        z = solve_z(f_adv, G, Enc)

        d_batch = 32
        for d_epoch in range(10):
            real_idx = mnp.random.randint(0, bs, (d_batch,))
            real = G(z_avg[real_idx]).clone().detach()
            fake_idx = mnp.random.randint(0, bs, (d_batch,))
            fake = G(z[fake_idx]).clone().detach()
            neg_logit = netD(fake)
            pos_logit = netD(real)

            gp = gradient_penalty(real, fake)
            l_gp = lambda_gp if gp <= 1 else ms.Tensor(40.0, ms.float32)
            EMD = ops.ReduceMean()(neg_logit) - ops.ReduceMean()(pos_logit)
            d_loss = EMD + l_gp * gp
            p_list.append(ops.ReduceMean()(EMD).asnumpy())
            print(f" gp {gp.asnumpy()}, d loss {d_loss.asnumpy()} ")
            d_optimizer.clear_grad()
            # TODO: MindSpore 反向传播 API
            d_loss.backward()
            d_optimizer.step()
            del fake, real

        inv_loss = f_loss(f_adv, Enc(G(z))) * 2000
        l_p = -ops.ReduceMean()(netD(G(z)))
        l_u = f_loss(f_adv, f0)
        tot_loss = ops.ReduceMean()(100 * l_p + 100 * beta * l_u)
        print(f'privacy loss is {l_p.asnumpy()}, utility loss is {l_u.asnumpy()}')
        hyper_grads = hypergradient(tot_loss, inv_loss, f_adv, z)
        f_optimizer.clear_grad()
        f_adv.grad = hyper_grads[0]
        print(f'check feature grad is {ops.Norm()(f_adv.grad)}')
        f_optimizer.step()
        u_list.append(l_u.asnumpy())

        if epoch % 3 == 0:
            save_tensor_images(G(z).detach(), f'{result_dir}/images/inv_{epoch}.jpg')
            ms.save_checkpoint(f_adv, f'{result_dir}/features/{batch_id}_{epoch}.ckpt')

    plot_loss(p_list, result_dir + '/curves', f'privacy_{batch_id}', 'EM distance')
    plot_loss(u_list, result_dir + '/curves', f'utility_{batch_id}', 'feature norm')



def hypergradient(tot_loss: ms.Tensor, inv_loss: ms.Tensor, f_adv: ms.Tensor, z: ms.Tensor):
    grad_op = ops.GradOperation(get_all=True, sens_param=False)
    v1 = grad_op(lambda x: tot_loss, z)
    for V in v1:
        print(f'EMD to z grad is {ops.Norm()(V)}')
    d_inv_d_z = grad_op(lambda x: inv_loss, z)
    v2 = approxInverseHVP(v1, d_inv_d_z, z, i=i, alpha=alpha)
    v3 = grad_op(lambda x: d_inv_d_z, f_adv, grad_outputs=v2, retain_graph=True)
    d_tot_d_f = grad_op(lambda x: tot_loss, f_adv)
    return [d - v for d, v in zip(d_tot_d_f, v3)]


def approxInverseHVP(v: Tuple[ms.Tensor], f: Tuple[ms.Tensor], z: ms.Tensor, i=30, alpha=0.01):
    p = v
    for j in range(i):
        grad_op = ops.GradOperation(get_all=True, sens_param=False)
        grad_v = grad_op(lambda x: f, z)
        if j % 30 == 0:
            print(f'inner epoch {j}, grad is {ops.Norm()(grad_v[0])}, v is {ops.Norm()(v[0])}')
        v = [v_ - alpha * g for v_, g in zip(v, grad_v)]
        p = [p_ + v_ for p_, v_ in zip(p, v)]
    return [alpha * p_ for p_ in p]


if __name__ == "__main__":
    figurePath = "../dataset/private"
    attributePath = "../dataset/eval_test.txt"
    bs = 128
    epochs = 450
    zd = 500
    lambda_gp = 20
    beta_list = [10]
    flr, alpha = 0.01, 0.001
    i_list = [120]

    _, dataloader = init_dataloader(attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1)

    G = newGanGenerator(in_dim=zd)
    G.load_state_dict(ms.load_checkpoint(f'../Gan/zdim_{zd}/params/G_90.ckpt'))
    G.set_train(False)

    D = GanDiscriminator()
    D.load_state_dict(ms.load_checkpoint(f'../Gan/zdim_{zd}/params/D_90.ckpt'))
    D.set_train(False)

    Enc = Encoder()
    Enc.model.fc = nn.Dense(512, 40)
    Enc.load_state_dict(ms.load_checkpoint('../params/enc.ckpt'))
    Enc.set_train(False)

    Amor = Amortizer(nz=zd)
    Amor.load_state_dict(ms.load_checkpoint('../params/amor.ckpt'))
    Amor.set_train(False)

    Dec = Decoder()
    Dec.load_state_dict(ms.load_checkpoint('../params/dec.ckpt'))
    Dec.set_train(False)

    for beta in beta_list:
        for i in i_list:
            result_dir = f'../Crafter_result/beta{beta}/'
            os.makedirs(result_dir, exist_ok=True)
            for sub in ['original', 'images', 'features', 'curves']:
                os.makedirs(os.path.join(result_dir, sub), exist_ok=True)
            for batch_id, (imgs, label) in enumerate(dataloader.create_tuple_iterator()):
                ms.ops.clear_cache()
                save_tensor_images(imgs.asnumpy(), os.path.join(result_dir, 'original', f'{batch_id}.jpg'))
                f0 = Enc(imgs)
                netD = GanDiscriminator()
                netD.load_state_dict(ms.load_checkpoint(f'../Gan/zdim_{zd}/params/D_90.ckpt'))
                gradient_based_ho(f0, netD, batch_id)
