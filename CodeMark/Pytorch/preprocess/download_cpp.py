from pprint import pprint

from datasets import load_dataset, Dataset, load_from_disk
import os
import json


def download(sample_num: int):
    os.environ['http_proxy'] = '127.0.0.1:7890'
    os.environ['https_proxy'] = '127.0.0.1:7890'

    dataset = load_dataset(
        "bigcode/the-stack-dedup",
        data_dir="data/cpp",
        split="train",
        streaming=True,
        token=""
    )

    # 获取前100条数据
    samples = []
    for i, sample in enumerate(dataset):
        samples.append(sample)
        if i > sample_num:
            break

    # 转为可缓存的 Dataset，并保存到本地
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(f"./cpp_{sample_num}_samples")


def main():
    # download(100)
    dataset = load_from_disk("cpp_100_samples")
    for i in range(10):
        print('='*32)
        print(dataset[i]['content'])


if __name__ == '__main__':
    main()
