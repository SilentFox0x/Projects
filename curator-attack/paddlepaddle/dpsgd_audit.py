import csv
import itertools

import paddle
import paddlenlp
from private_transformers import PrivacyEngine

eps = 1.0
delta = 0.0001
batch_size = 10
per_device_batch_size = 10
assert batch_size % per_device_batch_size == 0
grad_accumulate_num = batch_size // per_device_batch_size
audit_res_file = "audit_res_t5privtranstesttest.csv"
>>>>>>ds = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
text = ds["train"]["text"]
print(text[1])
model_name = "allenai/t5-small-next-word-generator-qoogle"
>>>>>>tokenizer = transformers.T5Tokenizer.from_pretrained(model_name, local_files_only=True)
>>>>>>model = transformers.T5ForConditionalGeneration.from_pretrained(
    model_name, local_files_only=True
).cuda()
vocab = tokenizer.get_vocab()
sequences = []
for se in text:
    word_list = se.split()
    if len(word_list) < 10:
        continue
    extra_id = vocab.get("<extra_id_0>")
    end = vocab.get("</s>")
    se_ids = tokenizer.encode(se, truncation=True, max_length=10)
    sequences.append(
        (
            paddle.to_tensor(data=se_ids[:-2] + [extra_id, end]),
            paddle.to_tensor(data=[extra_id, se_ids[-2], end]),
        )
    )


class TextDataset(paddle.io.Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return sequences[idx]


dataset = TextDataset(sequences)
dataloader = paddle.io.DataLoader(
    dataset=dataset, batch_size=per_device_batch_size, shuffle=True
)
dataloader2 = paddle.io.DataLoader(
    dataset=dataset, batch_size=per_device_batch_size, shuffle=True
)
embedding_dim = 10
hidden_dim = 20
num_layers = 8
observe_num = 10000
criterion = paddle.nn.CrossEntropyLoss()
optimizer = paddle.optimizer.SGD(
    parameters=model.parameters(), learning_rate=0.01, weight_decay=0.0
)
num_epochs = 10
from fastDP.accounting import accounting_manager

############################## 相关utils函数，如下 ##############################

def device2int(device):
    if isinstance(device, str):
        device = device.replace('cuda', 'gpu')
        device = device.replace('gpu:', '')
    return int(device)
############################## 相关utils函数，如上 ##############################


manager = accounting_manager.RDPManager(alphas=accounting_manager.DEFAULT_ALPHAS)
sigma = manager.compute_sigma(eps, delta, sample_rate=1, steps=1)
model.train()
privacy_engine = PrivacyEngine(
    model,
    batch_size=batch_size,
    sample_size=len(dataset),
    epochs=num_epochs,
    max_grad_norm=1.0,
    noise_multiplier=sigma,
)
privacy_engine.attach(optimizer)
o_list, o_prime_list = [], []
train_loader_iter = None
train_loader2_iter = None
total_step = observe_num + 2
cur_step = 0
model.train()
step = 0
while step < observe_num and cur_step < total_step:
    cur_step += 1
    success = 1
    optimizer.clear_gradients(set_to_zero=False)
    del train_loader_iter
    del train_loader2_iter
    train_loader_iter = iter(dataloader)
    train_loader2_iter = iter(dataloader2)
    epoch_finish = False
    while not epoch_finish and step < observe_num:
        optimizer.clear_gradients(set_to_zero=False)
        try:
            inputs, targets = next(train_loader_iter)
        except StopIteration:
            epoch_finish = True
            continue
        outputs = model(input_ids=inputs.cuda(), labels=targets.cuda())
        loss = outputs.loss
        optimizer.virtual_step(loss=loss)
        if epoch_finish:
            break
        param = next(itertools.islice(model.named_parameters(), 2, 3))
        print("auditing param:", param[0])
        k = 0
        o_prime = param[1].summed_grad.flatten()[k]
        o_prime += 1 / batch_size
        optimizer.clear_gradients(set_to_zero=False)
        del loss
        del outputs
        del inputs
        del targets
        try:
            inputs, targets = next(train_loader2_iter)
        except StopIteration:
            epoch_finish = True
            continue
        outputs = model(input_ids=inputs.cuda(), labels=targets.cuda())
        loss = outputs.loss
        optimizer.step(loss=loss)
        if epoch_finish:
            break
        param = next(itertools.islice(model.named_parameters(), 2, 3))
        print("auditing param:", param[0])
        k = 0
        o = param[1].summed_grad.flatten()[k]
        del loss
        del outputs
        del inputs
        del targets
        optimizer.clear_gradients(set_to_zero=False)
        if success:
            step += 1
            o_list.append(o.item())
            o_prime_list.append(o_prime.item())
            with open(audit_res_file, "a") as file:
                writer = csv.writer(file)
                writer.writerow([o.item(), o_prime.item()])
            del o
            del o_prime