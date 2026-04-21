import sys

sys.path.append("/home/liwei/tmp/paddle_project")
import os
import re
import time

import numpy as np
import paddle
import PIL
from paddle_utils import *
from PIL import Image


def init_dataloader(
    file_path,
    img_path,
    action="prt",
    batch_size=64,
    n_classes=1000,
    attriID=1,
    shuffle=False,
    skiprows=1,
    allAttri=False,
    normalization=False,
    stream=False,
):
    tf = time.time()
    data_set = ImageFolder(
        file_path,
        img_path,
        n_classes,
        attriID,
        skiprows,
        action,
        allAttri,
        normalization,
        stream,
    )
    data_loader = paddle.io.DataLoader(
        dataset=data_set, batch_size=batch_size, shuffle=shuffle, num_workers=0
    )
    interval = time.time() - tf
    return data_set, data_loader


class ImageFolder(paddle.io.Dataset):
    def __init__(
        self,
        file_path,
        img_path,
        n_classes,
        attriID,
        skiprows=1,
        action="prt",
        allAttri=False,
        normalization=False,
        stream=False,
    ):
        self.img_path = img_path
        self.allAttri = allAttri
        self.stream = stream
        self.trans = (
            self.get_processor()
            if normalization
            else paddle.vision.transforms.ToTensor()
        )
        self.img_list = os.listdir(self.img_path)
        self.action = action
        self.name_list, self.label_list = self.get_list(file_path, attriID, skiprows)
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = n_classes

    def get_list(self, file_path, attriID, skiprows=1):
        name_list, label_list = [], []
        f = open(file_path, "r", encoding="utf-8-sig")
        for _ in range(skiprows):
            f.readline()
        i = 0
        for line in f.readlines():
            item = line.strip()
            item = re.split(",|\\s ", item)
            img_name = item[0]
            if self.allAttri:
                iden = item[1:]
            else:
                iden = item[attriID]
            if self.action == "eval":
                if i == 0:
                    print("Loading inverted images for eval acc")
                for i in range(3):
                    name_list.append(f"{i}_{img_name}")
                    label_list.append(int(iden))
            elif self.action == "eval_fsim":
                if i == 0:
                    print("Loading original images for eval feature sim")
                for i in range(3):
                    name_list.append(img_name)
                    label_list.append(int(iden))
            else:
                if self.action == "inv_fawkes":
                    if i == 0:
                        print("Loading fawkes protected images for inversion")
                    img_name = img_name[:-4] + "_cloaked.png"
                elif self.action == "inv_lowkey":
                    if i == 0:
                        print("Loading lowkey protected images for inversion")
                    img_name = img_name[:-4] + "_attacked.png"
                elif (
                    self.action == "inv_ours"
                    or self.action == "inv_black"
                    or self.action == "inv_unprotected"
                ):
                    if i == 0:
                        print("Loading original images for inversion")
                elif self.action == "prt":
                    if i == 0:
                        print("Loading original images for protection")
                elif self.action == "utility_lowkey":
                    if i == 0:
                        print("Loading lowkey protected images for utility test")
                    img_name = img_name[:-4] + "_attacked.png"
                elif self.action == "utility_fawkes":
                    if i == 0:
                        print("Loading fawkes images for utility test")
                    img_name = img_name[:-4] + "_cloaked.png"
                elif i == 0:
                    print("Loading original images for utility test")
                name_list.append(img_name)
                if self.allAttri:
                    label_list.append(list(map(int, iden)))
                else:
                    label_list.append(int(iden))
            i = 1
        return name_list, paddle.to_tensor(data=label_list, dtype="float32")

    def load_img(self):
        if not self.stream:
            img_list = []
            for i, img_name in enumerate(self.name_list):
                path = self.img_path + "/" + img_name
                img = PIL.Image.open(path)
                img = img.convert("RGB")
                img_list.append(img)
            return img_list
        else:
            img_list = []
            for i, img_name in enumerate(self.name_list):
                path = self.img_path + "/" + img_name
                img_list.append(path)
            return img_list

    def get_processor(self):
        proc = []
        proc.append(paddle.vision.transforms.ToTensor())
        proc.append(
            paddle.vision.transforms.Normalize(
                mean=[0.4875, 0.4039, 0.3472], std=[0.156, 0.1401, 0.1372]
            )
        )
        return paddle.vision.transforms.Compose(transforms=proc)

    def __getitem__(self, index):
        if not self.stream:
            img = self.trans(self.image_list[index])
        else:
            img = self.image_list[index]
        label = self.label_list[index]
        return img, label

    def __len__(self):
        return self.num_img


def save_tensor_images(images, filename, nrow=None, normalize=True):
    if not nrow:
>>>>>>        torchvision.utils.save_image(images, filename, normalize=normalize, padding=0)
    else:
>>>>>>        torchvision.utils.save_image(
            images, filename, normalize=normalize, nrow=nrow, padding=0
        )


def transform_img_size(fake):
    bs = tuple(fake.shape)[0]
    fake_img = paddle.zeros(shape=(bs, 3, 64, 64))
    for i in range(bs):
>>>>>>        img_tmp = torchvision.transforms.ToPILImage()(fake[i].cpu())
        fake_tmp = paddle.vision.transforms.ToTensor()(img_tmp.resize((64, 64)))
        fake_img[i] = fake_tmp
    return fake_img


def find_most_id(identity, bs):
    identity, _ = paddle.sort(x=identity), paddle.argsort(x=identity)
    max_score = 0
    index = 0
    for i in range(1001):
        score = 0
        for j in range(bs):
            if identity[j] == i:
                score = score + 1
        if score >= max_score:
            max_score = score
            index = i
    return index


def load_img(image_path):
    img_list = []
    name_list = os.listdir(image_path)
    for img_name in enumerate(name_list):
        path = image_path + "/" + img_name[1]
        img = Image.open(path)
        img = paddle.unsqueeze(x=paddle.vision.transforms.ToTensor()(img), axis=0)
        img_list.append(img)
    return img_list, img_name[0] + 1


def get_feature(T, img, index):
    feature = T.module.get_fea(img)
    feature = feature[index]
    feature = feature.view(feature.shape[0], -1)
    return feature


def freeze(net):
    for p in net.parameters():
        out_0 = p
        out_0.stop_gradient = not False
        out_0


def unfreeze(net):
    for p in net.parameters():
        out_1 = p
        out_1.stop_gradient = not True
        out_1
