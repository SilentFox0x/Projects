import sys
import os
import argparse
import mindspore as ms
import mindspore.nn as nn
from mindspore import context, ops
from mindspore.vision.models import resnet18
import loader
from plot_loss import plot_loss

# 设置 MindSpore 上下文为 PYNATIVE 模式并使用 GPU
context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")

parser = argparse.ArgumentParser()
parser.add_argument('-d', action='store_true', default=False, help='debug')
parser.add_argument('-whole', action='store_true', default=False, help='Train the whole resnet rather than FC only')
args = parser.parse_args()

pubpath = '../dataset/public'
attrpath = '../dataset/pub_attri.csv'
result_dir = '../params/enc/'

net = resnet18(pretrained=True)

if not args.whole:
    for p in net.get_parameters():
        p.requires_grad = False

net.fc = nn.Dense(in_channels=512, out_channels=40)
# PyTorch DataParallel & .cuda() 对应：MindSpore 单卡直接使用
# 保留变量结构，不作 DataParallelWrapper
if args.d:
    import pdb; pdb.set_trace()

if args.whole:
    optim = nn.Adam(net.get_parameters())
else:
    optim = nn.Adam(net.fc.trainable_params())

criteria = nn.BCELoss(reduction='mean')

_, pubLoader = loader.init_dataloader(attrpath, pubpath, batch_size=64, n_classes=2, skiprows=1, allAttri=True)

loss_list = []
net.set_train(True)

if args.d:
    import pdb; pdb.set_trace()
for eid in range(30):
    for bid, (img, label) in enumerate(pubLoader.create_tuple_iterator()):
        label = label.astype(ms.float32)
        pred = ops.Sigmoid()(net(img))
        optim.clear_grad()
        loss = criteria(pred, label)
        loss_list.append(loss.asnumpy())
        loss.backward()
        optim.step()
    plot_loss(loss_list, result_dir, 'trainLoss', 'training loss')
    if args.whole:
        ms.save_checkpoint(net, os.path.join(result_dir, f'param_epoch_{eid}.ckpt'))
    else:
        ms.save_checkpoint(net.fc, os.path.join(result_dir, f'param_epoch_{eid}.ckpt'))

if args.whole:
    ms.save_checkpoint(net, '../params/enc.ckpt')
else:
    ms.save_checkpoint(net.fc, '../params/enc.ckpt')
