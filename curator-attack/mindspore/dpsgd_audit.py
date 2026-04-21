import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore import Tensor, context
from mindspore.dataset import GeneratorDataset
from private_transformers import PrivacyEngine
from datasets import load_dataset
import itertools
import csv
from transformers import T5ForConditionalGeneration, T5Tokenizer
import numpy as np

# 1. 上下文配置
context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

eps = 1.0
delta = 1e-4
batch_size = 10
per_device_batch_size = 10
assert batch_size % per_device_batch_size == 0
grad_accumulate_num = batch_size // per_device_batch_size
audit_res_file = 'audit_res_t5privtranstesttest.csv'
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
text = ds['train']['text']
print(text[1])

# 2. 模型与分词器
model_name = "allenai/t5-small-next-word-generator-qoogle"
tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
model = T5ForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
model.set_train()

# 3. 构造序列
vocab = tokenizer.get_vocab()
extra_id = vocab['<extra_id_0>']
end = vocab['</s>']
sequences = []
for se in text:
    word_list = se.split()
    if len(word_list) < 10:
        continue
    se_ids = tokenizer.encode(se, truncation=True, max_length=10)
    inp = se_ids[:-2] + [extra_id, end]
    tgt = [extra_id, se_ids[-2], end]
    sequences.append((inp, tgt))

# 4. Dataset 与 DataLoader
class TextDataset:
    def __init__(self, sequences):
        self.sequences = sequences
    def __len__(self):
        return len(self.sequences)
    def __getitem__(self, idx):
        inp, tgt = self.sequences[idx]
        return Tensor(np.array(inp), ms.int32), Tensor(np.array(tgt), ms.int32)

dataset = TextDataset(sequences)

def DataLoader(dataset, batch_size, shuffle=True):
    ds_gen = GeneratorDataset(dataset, ["inputs", "targets"], shuffle=shuffle)
    return ds_gen.batch(batch_size)

dataloader = DataLoader(dataset, per_device_batch_size, shuffle=True)
dataloader2 = DataLoader(dataset, per_device_batch_size, shuffle=True)

# 5. 超参数、损失和优化器
embedding_dim = 10
hidden_dim = 20
num_layers = 8
observe_num = 10000

criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
optimizer = nn.SGD(model.trainable_params(), learning_rate=0.01)

# 6. 计算噪声倍增器
num_epochs = 10
from fastDP.accounting import accounting_manager
manager = accounting_manager.RDPManager(alphas=accounting_manager.DEFAULT_ALPHAS)
sigma = manager.compute_sigma(eps, delta, sample_rate=per_device_batch_size/len(dataset), steps=1)

# 7. 隐私引擎附加
privacy_engine = PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(dataset),
    epochs=num_epochs,
    max_grad_norm=1.0,
    noise_multiplier=sigma,
)
privacy_engine.attach(optimizer)

# 8. 训练与审计循环
o_list, o_prime_list = [], []
train_loader_iter = None
train_loader2_iter = None
total_step = observe_num + 2
cur_step = 0
model.set_train()
step = 0

grad_fn = ops.GradOperation(get_by_list=True)

while step < observe_num and cur_step < total_step:
    cur_step += 1
    success = 1

    optimizer.zero_grad()
    train_loader_iter = dataloader.create_dict_iterator()
    train_loader2_iter = dataloader2.create_dict_iterator()
    epoch_finish = False
    while not epoch_finish and step < observe_num:
        optimizer.zero_grad()
        try:
            batch = next(train_loader_iter)
            inputs, targets = batch["inputs"], batch["targets"]
        except StopIteration:
            epoch_finish = True
            continue

        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        grads = grad_fn(model, model.trainable_params())(loss)
        optimizer.virtual_step(loss=loss)

        if epoch_finish:
            break

        param = next(itertools.islice(model.trainable_params_and_names(), 2, 3))
        print("auditing param:", param[0])
        k = 0
        o_prime = param[1].summed_grad.flatten()[k]
        o_prime += 1.0 / batch_size
        optimizer.zero_grad()
        del loss, outputs, inputs, targets

        try:
            batch2 = next(train_loader2_iter)
            inputs, targets = batch2["inputs"], batch2["targets"]
        except StopIteration:
            epoch_finish = True
            continue

        outputs = model(input_ids=inputs, labels=targets)
        loss = outputs.loss
        optimizer.step(loss=loss)

        if epoch_finish:
            break

        param = next(itertools.islice(model.trainable_params_and_names(), 2, 3))
        print("auditing param:", param[0])
        k = 0
        o = param[1].summed_grad.flatten()[k]
        del loss, outputs, inputs, targets

        optimizer.zero_grad()

        if success:
            step += 1
            o_list.append(o.asnumpy().item())
            o_prime_list.append(o_prime.asnumpy().item())
            with open(audit_res_file, 'a') as file:
                writer = csv.writer(file)
                writer.writerow([o.asnumpy().item(), o_prime.asnumpy().item()])
            del o, o_prime