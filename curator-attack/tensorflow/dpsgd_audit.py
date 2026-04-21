import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from fastDP.accounting import accounting_manager
from datasets import load_dataset
from transformers import TFAutoModelForSeq2SeqLM, T5Tokenizer
import csv

# ================= Hyperparameters =================
eps = 1.0
in_delta = 1e-4
batch_size = 10
per_device_batch_size = 10
assert batch_size % per_device_batch_size == 0
audit_res_file = 'audit_res_t5privtranstesttest.csv'
observe_num = 10000
num_epochs = 10

# ================ Load dataset ====================
ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
texts = ds['train']['text']

# =============== Tokenizer & Model ================
model_name = "allenai/t5-small-next-word-generator-qoogle"
tokenizer = T5Tokenizer.from_pretrained(model_name, local_files_only=True)
model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name, local_files_only=True)

# ============= Prepare sequences =================
vocab = tokenizer.get_vocab()
extra_id = vocab['<extra_id_0>']
end_id = vocab['</s>']
sequences = []
for se in texts:
    ids = tokenizer.encode(se, truncation=True, max_length=10)
    if len(ids) < 3:
        continue
    inp = ids[:-2] + [extra_id, end_id]
    tgt = [extra_id, ids[-2], end_id]
    sequences.append((inp, tgt))

# ============ Build tf.data.Dataset =============
def gen():
    for inp, tgt in sequences:
        yield {"input_ids": tf.constant(inp, dtype=tf.int32)}, tf.constant(tgt, dtype=tf.int32)

output_signature = (
    {"input_ids": tf.TensorSpec(shape=(None,), dtype=tf.int32)},
    tf.TensorSpec(shape=(3,), dtype=tf.int32),
)

dataset = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
# two parallel loaders
dataset1 = dataset.shuffle(len(sequences)).batch(per_device_batch_size)
# clone loader for second pass
dataset2 = dataset1

# ========== Compute noise multiplier =============
manager = accounting_manager.RDPManager(alphas=accounting_manager.DEFAULT_ALPHAS)
sample_rate = per_device_batch_size / len(sequences)
sigma = manager.compute_sigma(eps, in_delta, sample_rate=sample_rate, steps=1)

# =============== DP Optimizer ====================
optimizer = DPKerasSGDOptimizer(
    l2_norm_clip=1.0,
    noise_multiplier=sigma,
    num_microbatches=per_device_batch_size,
    learning_rate=0.01
)

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# =============== Training loop ==================
o_list, o_prime_list = [], []
step = 0

with open(audit_res_file, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # iterate until reaching observe_num
    while step < observe_num:
        for (x1, y1), (x2, y2) in zip(dataset1, dataset2):
            # --- virtual (first) step ---
            with tf.GradientTape() as tape:
                logits = model(x1['input_ids'], labels=y1).logits
                loss = loss_fn(y1, logits)
            grads = tape.gradient(loss, model.trainable_variables)
            # audit o_prime: first gradient component + 1/batch
            g_flat = tf.reshape(grads[2], [-1])
            o_prime = g_flat[0].numpy() + 1.0 / batch_size
            # accumulate grads (virtual)
            optimizer._accumulate_gradients(grads)

            # --- actual (second) step ---
            with tf.GradientTape() as tape2:
                logits2 = model(x2['input_ids'], labels=y2).logits
                loss2 = loss_fn(y2, logits2)
            grads2 = tape2.gradient(loss2, model.trainable_variables)
            optimizer.apply_gradients(zip(grads2, model.trainable_variables))
            g2_flat = tf.reshape(grads2[2], [-1])
            o = g2_flat[0].numpy()

            # record
            writer.writerow([o, o_prime])
            o_list.append(o)
            o_prime_list.append(o_prime)

            step += 1
            if step >= observe_num:
                break