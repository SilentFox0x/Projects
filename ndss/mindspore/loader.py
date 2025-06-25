import os
import re
import time
import numpy as np
from PIL import Image
import mindspore as ms
from mindspore import ops
import mindspore.dataset as ds
from mindspore.dataset.transforms import Compose
from mindspore.dataset.vision import transforms as vision


def init_dataloader(file_path, img_path, action='prt', batch_size=64, n_classes=1000, attriID=1, shuffle=False, skiprows=1,
                    allAttri=False, normalization=False, stream=False):
    tf = time.time()

    data_set = ImageFolder(file_path, img_path, n_classes, attriID, skiprows, action, allAttri, normalization, stream)
    data_loader = ds.GeneratorDataset(data_set, ['img', 'label'], shuffle=shuffle)
    data_loader = data_loader.batch(batch_size)

    interval = time.time() - tf
    # print('Initializing data loader took %ds' % interval)
    return data_set, data_loader


class ImageFolder:
    def __init__(self, file_path, img_path, n_classes, attriID, skiprows=1, action='prt', allAttri=False, normalization=False, stream=False):
        self.img_path = img_path
        self.allAttri = allAttri
        self.stream = stream
        self.trans = self.get_processor() if normalization else vision.ToTensor()
        self.img_list = os.listdir(self.img_path)
        self.action = action
        self.name_list, self.label_list = self.get_list(file_path, attriID, skiprows)
        self.image_list = self.load_img()
        self.num_img = len(self.image_list)
        self.n_classes = n_classes
        # print("Load " + str(self.num_img) + " images")

    def get_list(self, file_path, attriID, skiprows=1):
        name_list, label_list = [], []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for _ in range(skiprows):
                f.readline()
            i = 0
            for line in f.readlines():
                item = line.strip()
                item = re.split(r',|\s ', item)
                img_name = item[0]
                if self.allAttri:
                    iden = item[1:]
                else:
                    iden = item[attriID]
                if self.action == 'eval':
                    if i == 0:
                        print('Loading inverted images for eval acc')
                    for k in range(3):
                        name_list.append(f'{k}_{img_name}')
                        label_list.append(int(iden))
                elif self.action == 'eval_fsim':
                    if i == 0:
                        print('Loading original images for eval feature sim')
                    for k in range(3):
                        name_list.append(img_name)
                        label_list.append(int(iden))
                else:
                    if self.action == 'inv_fawkes':
                        if i == 0:
                            print('Loading fawkes protected images for inversion')
                        img_name = img_name[:-4] + '_cloaked.png'
                    elif self.action == 'inv_lowkey':
                        if i == 0:
                            print('Loading lowkey protected images for inversion')
                        img_name = img_name[:-4] + '_attacked.png'
                    elif self.action in ['inv_ours', 'inv_black', 'inv_unprotected']:
                        if i == 0:
                            print('Loading original images for inversion')
                    else:
                        if self.action == 'prt':
                            if i == 0:
                                print('Loading original images for protection')
                        elif self.action == 'utility_lowkey':
                            if i == 0:
                                print('Loading lowkey protected images for utility test')
                            img_name = img_name[:-4] + '_attacked.png'
                        elif self.action == 'utility_fawkes':
                            if i == 0:
                                print('Loading fawkes images for utility test')
                            img_name = img_name[:-4] + '_cloaked.png'
                        else:
                            if i == 0:
                                print('Loading original images for utility test')

                    name_list.append(img_name)
                    if self.allAttri:
                        label_list.append(list(map(int, iden)))
                    else:
                        label_list.append(int(iden))
                i = 1

        return name_list, ms.Tensor(label_list, ms.float32)

    def load_img(self):
        if not self.stream:
            img_list = []
            for img_name in self.name_list:
                path = f"{self.img_path}/{img_name}"
                img = Image.open(path).convert('RGB')
                img_list.append(img)
            return img_list
        else:
            img_list = []
            for img_name in self.name_list:
                path = f"{self.img_path}/{img_name}"
                img_list.append(path)
            return img_list

    def get_processor(self):
        proc = [
            vision.ToTensor(),
            vision.Normalize(mean=[0.4875, 0.4039, 0.3472], std=[0.1560, 0.1401, 0.1372])
        ]
        return Compose(proc)

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
    import math
    imgs = images.asnumpy() if isinstance(images, ms.Tensor) else np.array(images)
    bs, c, h, w = imgs.shape
    if not nrow:
        nrow = bs
    grid = Image.new('RGB', (w * nrow, h * math.ceil(bs / nrow)))
    for idx in range(bs):
        row = idx // nrow
        col = idx % nrow
        img = (imgs[idx].transpose(1, 2, 0) * 255).astype(np.uint8)
        pil_img = Image.fromarray(img)
        grid.paste(pil_img, (col * w, row * h))
    grid.save(filename)


def transform_img_size(fake):
    bs = fake.shape[0]
    fake_img = ms.Tensor(np.zeros((bs, 3, 64, 64)), ms.float32)
    for i in range(bs):
        arr = fake[i].asnumpy().transpose(1, 2, 0)
        pil = Image.fromarray((arr * 255).astype(np.uint8))
        resized = pil.resize((64, 64))
        tmp = np.array(resized).transpose(2, 0, 1) / 255.0
        fake_img[i] = ms.Tensor(tmp, ms.float32)
    return fake_img


def find_most_id(identity, bs):
    identity, _ = ops.Sort()(identity)
    max_score = 0
    index = 0
    for i in range(1001):
        score = int((identity == i).sum().asnumpy())
        if score >= max_score:
            max_score = score
            index = i
    return index


def load_img(image_path):
    img_list = []
    name_list = os.listdir(image_path)
    for img_name in enumerate(name_list):
        path = f"{image_path}/{img_name[1]}"
        img = Image.open(path).convert('RGB')
        tensor = vision.ToTensor()(img)
        tensor = ops.ExpandDims()(tensor, 0)
        img_list.append(tensor)
    return img_list, img_name[0] + 1


def get_feature(T, img, index):
    feature = T.module.get_fea(img)
    feature = feature[index]
    feature = feature.reshape(feature.shape[0], -1)
    return feature


def freeze(net):
    for p in net.get_parameters():
        p.requires_grad = False


def unfreeze(net):
    for p in net.get_parameters():
        p.requires_grad = True
