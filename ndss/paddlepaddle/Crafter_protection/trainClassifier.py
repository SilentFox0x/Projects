import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import os
import random
import sys

import paddle
from paddle_utils import *

sys.path.append("..")
from loader import *
from models import *

PRT = "IFT"
bs = 128
max_epoch = 50
num_class = 40
for beta in [None]:
    mode = f"beta{beta}"
    figurePath = "../dataset/private"
    attributePath = "../dataset/attri_test.txt"
    result_dir = f"../params/cf/"
    os.makedirs(result_dir, exist_ok=True)
    print("Loading images from:", figurePath)
    print("TrainCF")
    print("Result Dir:", result_dir)
    _, dataloader = init_dataloader(
        attributePath,
        figurePath,
        action="prt",
        batch_size=bs,
        n_classes=num_class,
        skiprows=1,
        allAttri=True,
    )
    Enc = Encoder()
    Enc = paddle.DataParallel(layers=Enc).cuda()
    Enc.eval()
    cf = Classifier(num_class)
    cf = paddle.DataParallel(layers=cf).cuda()
    cf.train()
    opt = paddle.optimizer.Adam(parameters=cf.parameters(), weight_decay=0.0)
    loss_fn = paddle.nn.CrossEntropyLoss()
    loss_list = []
    for e in range(max_epoch):
        print("Epoch", e)
        for i, (img, label) in enumerate(dataloader):
            img = img.cuda()
            label = label.cuda()
            with paddle.no_grad():
                feature = Enc(img)
            opt.clear_gradients(set_to_zero=False)
            pred = cf(feature)
            loss = loss_fn(pred, label)
            loss.backward()
            loss_list.append(loss.item())
            opt.step()
        paddle.save(
            obj=cf.module.state_dict(), path=os.path.join(result_dir, f"cf_{e}.pkl")
        )
