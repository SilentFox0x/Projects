# CodeMark

This repo provides the code for reproducing the experiments for code watermark. 

## Requirements

* [Python3](https://www.linuxbabe.com/ubuntu/install-python-3-6-ubuntu-16-04-16-10-17-04) 
* [PyTorch](https://pytorch.org/)
* `pip install -r requirements.txt`
* Build tree-sitter. We use tree-sitter to parse code snippets and construct graphs. Follow the steps below to build a parser for `tree-sitter`.

```shell
# create a directory to store sources
mkdir tree-sitter
cd tree-sitter

# clone parser repositories
git clone https://github.com/tree-sitter/tree-sitter-java.git
cd tree-sitter-java
git checkout 6c8329e2da78fae78e87c3c6f5788a2b005a4afc
cd ..

git clone https://github.com/tree-sitter/tree-sitter-cpp.git
cd tree-sitter-cpp
git checkout 0e7b7a02b6074859b51c1973eb6a8275b3315b1d
cd ..

git clone https://github.com/tree-sitter/tree-sitter-javascript.git
cd tree-sitter-javascript
git checkout f772967f7b7bc7c28f845be2420a38472b16a8ee
cd ..

git clone https://github.com/tree-sitter/tree-sitter-python.git
cd tree-sitter-python
git checkout 0f9047c
cd ..

# go back to parent dir
cd ..

# run python script to build the parser
python build_treesitter_langs.py ./tree-sitter

# the built parser will be put under ./resources/my-languages.so
```

## Quick Start

### 1. Prepare the dataset

The CSN datasets are available on the project site of [CodeSearchNet](https://github.com/github/CodeSearchNet). Since CSN datasets are relatively large, they are not included here. Follow the steps below to further process the dataset after downloading it.

1. Follow the instructions on [CodeXGLUE (code summarization task)](https://github.com/microsoft/CodeXGLUE/tree/main/Code-Text/code-to-text) to filter the dataset.
2. pre-process the dataset to filter out samples with grammar errors or unsupported features.

```sh
python prepare.py --dataset_dir ./codesearchnet
```

The results will be stored as `filename_filtered.jsonl`, rename it into `train.jsonl` or `valid.jsonl` or `test.jsonl` depending on the split and put the three files under `./datasets/csn_java` or `./datasets/csn_js`.

3. After all datasets are processed, the final directory should look like this

```txt
- datasets
    - csn_java
        - train.jsonl
        - valid.jsonl
        - test.jsonl
    - csn_js
        - train.jsonl
        - valid.jsonl
        - test.jsonl
```


### 2. Train 

* All the configurations set in `./configs/my_model.yaml`.
* `train.py` is responsible for all training tasks. Refer to the `parse_args()` function in `train.py` for more details on the arguments.

Here are some examples.

```sh
# training a 4-bit model for Java on GPU:0
python train.py \
    --lang=java \
    --n_bits=4 \
    --epochs=10 \
    --batch_size 64 \
    --shared_encoder \
    --device 0 \
    --substitute_mask_probability 0.25
```


### 3. Inference
* embed watermark

`python embedder.py --model_path codemark_model.pth --input ./codesearchnet/test.jsonl --output ./codesearchnet/watermarked-test.jsonl` 

* extract watermark

`python extractor.py --model_path codemark_model.pth --input ./codesearchnet/watermarked-test.jsonl --output ./codesearchnet/extracted-test.jsonl`

### 4. Support Four Deep Learning Frameworks
This repo supports four deep learning frameworks: PyTorch, TensorFlow, PaddlePaddle, and MindSpore. You can find the models of different frameworks in `model` folder.
