import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import os
import sys

import paddle
from paddle_utils import *

sys.path.append("..")
from typing import Tuple

from loader import *
from models import *
from tqdm import tqdm
from WhiteBoxAttack.attacker import *


def f_loss(f, f_rec):
    n, d = len(f), 64 * 16 * 16
    mse = paddle.linalg.norm(x=f.view(n, -1) - f_rec.view(n, -1), axis=1) ** 2
    mse = mse / d
    return paddle.mean(x=mse)


def solve_z(feature, G, Enc):
    """
    Find initial x* of feature
    """
    G.clear_gradients(set_to_zero=False)
    Enc.clear_gradients(set_to_zero=False)
    print("with amortize")
    z = amor_inversion(G, D, Enc, Amor, feature, z_dim=zd)
    out_3 = z.clone().detach()
    out_3.stop_gradient = not True
    z = out_3
    G.clear_gradients(set_to_zero=False)
    Enc.clear_gradients(set_to_zero=False)
    return z


def gradient_penalty(x, y):
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    alpha_grad = paddle.rand(shape=shape).cuda()
    interpolate = x + alpha_grad * (y - x)
    interpolate = interpolate.cuda()
    interpolate.stop_gradient = not True
    o = netD(interpolate)
    g = grad(
        o,
        interpolate,
        grad_outputs=paddle.ones(shape=tuple(o.shape)).cuda(),
        create_graph=True,
    )[0].view(interpolate.shape[0], -1)
    gp = ((g.norm(p=2, axis=1) - 1) ** 2).mean()
    return gp


def gradient_based_ho(f0, netD, batch_id):
    out_4 = f0.clone().detach()
    out_4.stop_gradient = not True
    f_adv = out_4.cuda()
    f_optimizer = paddle.optimizer.Adam(
        parameters=[f_adv], learning_rate=flr, weight_decay=0.0
    )
    d_optimizer = paddle.optimizer.Adam(
        parameters=netD.parameters(), learning_rate=0.0005, weight_decay=0.0
    )
    p_list, u_list = [], []
    z_avg = paddle.randn(shape=[bs, zd]).cuda().astype(dtype="float32")
    for epoch in range(epochs):
        print(f"\nepoch {epoch}")
        z = solve_z(f_adv, G, Enc)
        d_batch = 32
        for d_epoch in range(10):
            real_index = paddle.randint(low=0, high=bs, shape=(d_batch,))
            real = G(z_avg[real_index]).clone().detach()
            fake_index = paddle.randint(low=0, high=bs, shape=(d_batch,))
            fake = G(z[fake_index]).clone().detach()
            neg_logit = netD(fake)
            pos_logit = netD(real)
            gp = gradient_penalty(real, fake)
            l_gp = lambda_gp
            if gp > 1:
                l_gp = 40
            EMD = neg_logit.mean() - pos_logit.mean()
            d_loss = EMD + l_gp * gp
            p_list.append(paddle.mean(x=EMD).data.cpu())
            print(" gp {}, d loss {} ".format(gp.item(), d_loss.item()))
            d_optimizer.clear_gradients(set_to_zero=False)
            d_loss.backward()
            d_optimizer.step()
            del fake
            del real
        inv_loss = f_loss(f_adv, Enc(G(z))) * 2000
        l_p = -netD(G(z)).mean()
        l_u = f_loss(f_adv, f0)
        tot_loss = paddle.mean(x=100 * l_p + 100 * beta * l_u)
        print(f"privacy loss is {l_p}, utility loss is {l_u}")
        hyper_grads = hypergradient(tot_loss, inv_loss, f_adv, z)
        f_optimizer.clear_gradients(set_to_zero=False)
        f_adv.grad = hyper_grads[0]
        print(f"check feature grad is {paddle.linalg.norm(x=f_adv.grad)}")
        f_optimizer.step()
        u_list.append(l_u.data.cpu())
        if epoch % 3 == 0:
            save_tensor_images(G(z).detach(), f"{result_dir}/images/inv_{epoch}.jpg")
            paddle.save(
                obj=f_adv.detach(), path=f"{result_dir}/features/{batch_id}_{epoch}.pt"
            )
    plot_loss(
        p_list, result_dir + "/curves", "privacy_{}".format(batch_id), "EM distance"
    )
    plot_loss(
        u_list, result_dir + "/curves", "utility_{}".format(batch_id), "feature norm"
    )
    return


def hypergradient(
    tot_loss: paddle.Tensor,
    inv_loss: paddle.Tensor,
    f_adv: paddle.Tensor,
    z: paddle.Tensor,
):
    v1 = paddle.grad(outputs=tot_loss, inputs=z, retain_graph=True)
    for V in v1:
        print(f"EMD to z grad is {paddle.linalg.norm(x=V)}")
    d_inv_d_z = paddle.grad(outputs=inv_loss, inputs=z, create_graph=True)
    v2 = approxInverseHVP(v1, d_inv_d_z, z, i=i, alpha=alpha)
    v3 = paddle.grad(
        outputs=d_inv_d_z, inputs=f_adv, grad_outputs=v2, retain_graph=True
    )
    d_tot_d_f = paddle.grad(outputs=tot_loss, inputs=f_adv)
    return [(d - v) for d, v in zip(d_tot_d_f, v3)]


def approxInverseHVP(
    v: paddle.Tensor, f: paddle.Tensor, z: paddle.Tensor, i=30, alpha=0.01
):
    p = v
    for j in range(i):
        grad = paddle.grad(outputs=f, inputs=z, grad_outputs=v, retain_graph=True)
        if j % 30 == 0:
            print(
                f"inner epoch {j}, grad is {paddle.linalg.norm(x=grad[0])}, v is {paddle.linalg.norm(x=v[0])}"
            )
        v = [(v_ - alpha * g) for v_, g in zip(v, grad)]
        p = [(p_ + v_) for p_, v_ in zip(p, v)]
    p = [(alpha * p_) for p_ in p]
    return p


if __name__ == "__main__":
    device = "cuda"
    paddle.device.get_device()
>>>>>>    torch.cuda._initialized = True
    figurePath = "../dataset/private"
    attributePath = "../dataset/eval_test.txt"
    bs = 128
    epochs = 450
    zd = 500
    lambda_gp = 20
    beta_list = [10]
    flr, alpha = 0.01, 0.001
    i_list = [120]
    _, dataloader = init_dataloader(
        attributePath, figurePath, batch_size=bs, n_classes=2, attriID=1, skiprows=1
    )
    G = newGanGenerator(in_dim=zd)
    G.set_state_dict(
        state_dict=paddle.load(path=str("../Gan/zdim_{}/params/G_90.pkl".format(zd)))
    )
    G = paddle.DataParallel(layers=G).cuda()
    G.eval()
    D = GanDiscriminator()
    D.set_state_dict(
        state_dict=paddle.load(path=str("../Gan/zdim_{}/params/D_90.pkl".format(zd)))
    )
    D = paddle.DataParallel(layers=D).cuda()
    D.eval()
    Enc = Encoder()
    Enc.model.fc = paddle.nn.Linear(in_features=512, out_features=40)
    Enc.model.load_state_dict(paddle.load(path=str("../params/enc.pt")))
    Enc = paddle.DataParallel(layers=Enc).cuda()
    Enc.eval()
    Amor = Amortizer(nz=zd)
    Amor.set_state_dict(state_dict=paddle.load(path=str("../params/amor.pkl")))
    Amor = paddle.DataParallel(layers=Amor).cuda()
    Amor.eval()
    Dec = Decoder()
    Dec = paddle.DataParallel(layers=Dec).cuda()
    Dec.set_state_dict(state_dict=paddle.load(path=str("../params/dec.pkl")))
    Dec.eval()
    for beta in beta_list:
        for i in i_list:
            result_dir = f"../Crafter_result/beta{beta}/"
            os.makedirs(result_dir, exist_ok=True)
            os.makedirs(result_dir + "/original", exist_ok=True)
            os.makedirs(result_dir + "/images", exist_ok=True)
            os.makedirs(result_dir + "/features", exist_ok=True)
            os.makedirs(result_dir + "/curves", exist_ok=True)
            for batch_id, (imgs, label) in enumerate(dataloader):
                paddle.device.cuda.empty_cache()
                imgs = imgs.cuda()
                save_tensor_images(
                    imgs.detach(), result_dir + f"/original/{batch_id}.jpg"
                )
                f0 = Enc(imgs).detach()
                netD = GanDiscriminator()
                netD = paddle.DataParallel(layers=netD).cuda()
                gradient_based_ho(f0, netD, batch_id)
