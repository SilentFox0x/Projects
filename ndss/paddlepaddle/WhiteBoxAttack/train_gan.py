import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import argparse
import math
import os
import sys

import numpy as np
import paddle
from paddle_utils import *

sys.path.append("..")
import urllib3
from models import *

device = device2str("cuda:0" if paddle.device.cuda.device_count() >= 1 else "cpu")
gender_idx = 20
target_idx = gender_idx


def gradient_penalty(x, y):
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    alpha = paddle.rand(shape=shape).cuda()
    z = x + alpha * (y - x)
    z = z.cuda()
    z.stop_gradient = not True
    o = discriminator(z)
    g = paddle.grad(
        outputs=o,
        inputs=z,
        grad_outputs=paddle.ones(shape=tuple(o.shape)).cuda(),
        create_graph=True,
    )[0].view(z.shape[0], -1)
    gp = ((g.norm(p=2, axis=1) - 1) ** 2).mean()
    return gp


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=500,
        help="number of epochs of training [deprecated]",
    )
    parser.add_argument(
        "--n_iters", type=int, default=100000, help="number of iterations of training"
    )
    parser.add_argument(
        "--latent_dim", type=int, default=128, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--img_size", type=int, default=64, help="size of each image dimension"
    )
    parser.add_argument(
        "--data_num", type=int, default=4096, help="size of training dataset"
    )
    parser.add_argument(
        "--n_critic",
        type=int,
        default=5,
        help="number of training steps for discriminator per iter",
    )
    parser.add_argument("--lr", type=float, default=0.0004, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    opt = parser.parse_args()
    print(opt)
    latent_dim = opt.latent_dim
    bs = 64
    lambda_gp = 10
    print("latent dimension ", latent_dim)
    print("learning rate", opt.lr)
    generator = GanGenerator(in_dim=latent_dim)
    generator = paddle.DataParallel(layers=generator).cuda()
    discriminator = GanDiscriminator(in_dim=2)
    discriminator = paddle.DataParallel(layers=discriminator).cuda()
    trans_n = paddle.vision.transforms.Normalize(
        mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
    )
    trans_crop = paddle.vision.transforms.CenterCrop(size=128)
    trnas_resize = paddle.vision.transforms.Resize(size=64)
    trans_tensor = paddle.vision.transforms.ToTensor()
    trans = paddle.vision.transforms.Compose(
        transforms=[trans_crop, trnas_resize, trans_tensor]
    )
>>>>>>    dataset = torchvision.datasets.CelebA(
        "../dataset/",
        split="train",
        target_type="attr",
        transform=trans,
        target_transform=None,
        download=False,
    )
    paddle.seed(seed=0)
    length = len(dataset)
    dataset, _ = paddle.io.random_split(
        dataset=dataset, lengths=[opt.data_num, length - opt.data_num]
    )
    dataloader = paddle.io.DataLoader(dataset=dataset, batch_size=bs)
    result_dir = "../Gan/zdim_{}/images/".format(opt.latent_dim)
    os.makedirs(result_dir, exist_ok=True)
    store_param = "../Gan/zdim_{}/params/".format(opt.latent_dim)
    os.makedirs(store_param, exist_ok=True)
    optimizer_G = paddle.optimizer.Adam(
        parameters=generator.parameters(),
        learning_rate=opt.lr,
        beta1=(opt.b1, opt.b2)[0],
        beta2=(opt.b1, opt.b2)[1],
        weight_decay=0.0,
    )
    optimizer_D = paddle.optimizer.Adam(
        parameters=discriminator.parameters(),
        learning_rate=opt.lr,
        beta1=(opt.b1, opt.b2)[0],
        beta2=(opt.b1, opt.b2)[1],
        weight_decay=0.0,
    )
    Tensor = paddle.float32
    batches_done = 0
    epoch = 0
    while batches_done < opt.n_iters:
        for i, (imgs, _) in enumerate(dataloader):
            epoch += 1
            real_imgs = imgs.cuda()
            out_5 = paddle.randn(shape=(tuple(imgs.shape)[0], latent_dim))
            out_5.stop_gradient = not True
            z = out_5.cuda()
            fake_imgs = generator(z)
            real_validity = discriminator(real_imgs)
            fake_validity = discriminator(fake_imgs)
            gp = gradient_penalty(real_imgs.data, fake_imgs.data)
            d_loss = (
                -paddle.mean(x=real_validity)
                + paddle.mean(x=fake_validity)
                + lambda_gp * gp
            )
            optimizer_D.clear_gradients(set_to_zero=False)
            d_loss.backward()
            optimizer_D.step()
            optimizer_G.clear_gradients(set_to_zero=False)
            if (batches_done + 1) % opt.n_critic == 0:
                fake_imgs = generator(z)
                fake_validity = discriminator(fake_imgs)
                g_loss = -paddle.mean(x=fake_validity)
                g_loss.backward()
                optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [loss_div: %s]"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        d_loss.item(),
                        g_loss.item(),
                        "deprecated",
                    )
                )
                print("gp is {}".format(gp))
            batches_done += 1
            if batches_done % (opt.n_critic * 1000) == 0:
                paddle.save(
                    obj=generator.module.state_dict(),
                    path=store_param + "G_{}.pkl".format(batches_done),
                )
                paddle.save(
                    obj=discriminator.module.state_dict(),
                    path=store_param + "D_{}.pkl".format(batches_done),
                )
>>>>>>                torchvision.utils.save_image(
                    fake_imgs.detach().data,
                    result_dir + "/img_{}.jpg".format(batches_done),
                )
    paddle.save(
        obj=generator.module.state_dict(),
        path=store_param + "G_{}.pkl".format(batches_done),
    )
    paddle.save(
        obj=discriminator.module.state_dict(),
        path=store_param + "D_{}.pkl".format(batches_done),
    )
>>>>>>    torchvision.utils.save_image(
        fake_imgs.detach().data, result_dir + "/img_{}.jpg".format(batches_done)
    )
