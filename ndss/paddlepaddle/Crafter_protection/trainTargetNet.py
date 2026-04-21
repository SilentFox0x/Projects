import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import os
import sys

import paddle
from paddle_utils import *

sys.path.append("..")
import argparse

import loader
from plot_loss import plot_loss

parser = argparse.ArgumentParser()
parser.add_argument("-d", action="store_true", default=False, help="debug")
parser.add_argument(
    "-whole",
    action="store_true",
    default=False,
    help="Train the whole resnet rather than FC only",
)
args = parser.parse_args()
pubpath = "../dataset/public"
attrpath = "../dataset/pub_attri.csv"
result_dir = "../params/enc/"
net = paddle.vision.models.resnet18(pretrained=True)
if not args.whole:
    for p in net.parameters():
        out_2 = p
        out_2.stop_gradient = not False
        out_2
net.fc = paddle.nn.Linear(in_features=512, out_features=40)
net = paddle.DataParallel(layers=net).cuda()
if args.d:
    import pdb

    pdb.set_trace()
if args.whole:
    optim = paddle.optimizer.Adam(parameters=net.parameters(), weight_decay=0.0)
else:
    optim = paddle.optimizer.Adam(
        parameters=net.module.fc.parameters(), weight_decay=0.0
    )
criteria = paddle.nn.BCELoss()
_, pubLoader = loader.init_dataloader(
    attrpath, pubpath, batch_size=64, n_classes=2, skiprows=1, allAttri=True
)
loss_list = list()
net.train()
if args.d:
    import pdb

    pdb.set_trace()
for eid in range(30):
    for bid, (img, label) in enumerate(pubLoader):
        img = img.cuda()
        label = label.cuda()
        pred = paddle.nn.functional.sigmoid(x=net(img))
        optim.clear_gradients(set_to_zero=False)
        loss = criteria(pred, label)
        loss.backward()
        loss_list.append(loss.item())
        optim.step()
    plot_loss(loss_list, result_dir, "trainLoss", "training loss")
    if args.whole:
        paddle.save(
            obj=net.module.state_dict(),
            path=os.path.join(result_dir, "param_epoch_{}.pt".format(eid)),
        )
    else:
        paddle.save(
            obj=net.module.fc.state_dict(),
            path=os.path.join(result_dir, "param_epoch_{}.pt".format(eid)),
        )
if args.whole:
    paddle.save(obj=net.module.state_dict(), path="../params/enc.pt")
else:
    paddle.save(obj=net.module.fc.state_dict(), path="../params/enc.pt")
