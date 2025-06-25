import tensorflow as tf
import numpy as np
import os
import re
from PIL import Image


def init_dataloader_tf(file_path, img_path, action='prt', batch_size=64,
                       n_classes=1000, attriID=1, shuffle=False,
                       skiprows=1, allAttri=False, normalization=False,
                       stream=False):
    """
    Initialize TensorFlow Dataset from image folder and label list.
    Returns:
      - builder: ImageFolderTF instance
      - tf_dataset: tf.data.Dataset yielding (img, label)
    """
    builder = ImageFolderTF(file_path, img_path, n_classes, attriID,
                            skiprows, action, allAttri,
                            normalization, stream)
    ds = builder.get_tf_dataset(batch_size=batch_size, shuffle=shuffle)
    return builder, ds


class ImageFolderTF:
    def __init__(self, file_path, img_path, n_classes=1000, attriID=1,
                 skiprows=1, action='prt', allAttri=False,
                 normalization=False, stream=False):
        self.img_path = img_path
        self.allAttri = allAttri
        self.stream = stream
        self.normalization = normalization
        self.n_classes = n_classes
        self.action = action

        self.name_list, self.label_list = self._get_list(
            file_path, attriID, skiprows)
        self.image_list = self._load_img_paths()
        self.num_img = len(self.image_list)

    def _get_list(self, file_path, attriID, skiprows):
        name_list, label_list = [], []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for _ in range(skiprows):
                next(f)
            for i, line in enumerate(f):
                parts = re.split(r',|\s+', line.strip())
                img_name = parts[0]
                iden = parts[1:] if self.allAttri else parts[attriID]

                # Handle action-specific renaming
                if self.action.startswith('inv_') or self.action.startswith('utility_'):
                    if 'fawkes' in self.action:
                        img_name = img_name[:-4] + '_cloaked.png'
                    elif 'lowkey' in self.action:
                        img_name = img_name[:-4] + '_attacked.png'

                # eval cases: repeat entries
                if self.action in ('eval', 'eval_fsim'):
                    for j in range(3):
                        name_list.append(f'{j}_{img_name}' if self.action=='eval' else img_name)
                        label_list.append(int(iden))
                else:
                    name_list.append(img_name)
                    if self.allAttri:
                        label_list.append(list(map(int, iden)))
                    else:
                        label_list.append(int(iden))
        labels = tf.convert_to_tensor(label_list, dtype=tf.float32)
        return name_list, labels

    def _load_img_paths(self):
        # Returns list of file paths (no decoding yet)
        return [os.path.join(self.img_path, name) for name in self.name_list]

    def _process_image(self, path):
        img = Image.open(path).convert('RGB')
        img = tf.convert_to_tensor(np.array(img), dtype=tf.float32) / 255.0
        if self.normalization:
            mean = tf.constant([0.4875, 0.4039, 0.3472], shape=[1,1,3])
            std  = tf.constant([0.1560, 0.1401, 0.1372], shape=[1,1,3])
            img = (img - mean) / std
        return img

    def get_tf_dataset(self, batch_size=64, shuffle=False):
        def gen():
            for path, label in zip(self.image_list, self.label_list):
                img = self._process_image(path)
                yield img, label

        ds = tf.data.Dataset.from_generator(
            gen,
            output_signature=(
                tf.TensorSpec(shape=(None,None,3), dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.float32)
            )
        )
        if shuffle:
            ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        return ds


# Utility functions

def save_tensor_images(images, filename, nrow=None, normalize=True):
    """
    Save batch of images to files. If nrow not None, grid saving not supported.
    """
    if nrow:
        raise NotImplementedError('Grid saving with nrow not implemented.')
    imgs = images
    if normalize:
        imgs = tf.clip_by_value(imgs, 0.0, 1.0)
    bs = tf.shape(imgs)[0]
    for i in range(bs):
        img = imgs[i]
        # convert to uint8 PNG
        png = tf.image.encode_png(tf.image.convert_image_dtype(img, tf.uint8))
        base, ext = os.path.splitext(filename)
        out_path = filename if bs==1 else f"{base}_{i}{ext}"
        tf.io.write_file(out_path, png)


def transform_img_size(fake):
    """
    Resize input tensor to 64x64. Expects shape [bs, h, w, c].
    """
    return tf.image.resize(fake, [64,64])


def find_most_id_tf(identity, bs=None):
    """
    Given 1-D int tensor identity, return the most frequent value.
    """
    identity = tf.reshape(identity, [-1])
    unique, _, count = tf.unique_with_counts(identity)
    idx = tf.argmax(count)
    return unique[idx]


def freeze(module):
    """
    Freeze trainable variables of a tf.Module or keras layer.
    """
    if hasattr(module, 'trainable_variables'):
        for var in module.trainable_variables:
            try:
                var._trainable = False
            except AttributeError:
                pass


def unfreeze(module):
    """
    Unfreeze trainable variables of a tf.Module or keras layer.
    """
    if hasattr(module, 'trainable_variables'):
        for var in module.trainable_variables:
            try:
                var._trainable = True
            except AttributeError:
                pass