from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch
import poptorch
from ipu_options import get_options
from optimization import get_optimizer

from offline_config import InferDescription
from bert_model.modeling import PipelinedBertForSequenceClassification
import time
import numpy as np

inference_config = InferDescription()
inference_config.detail.batch_size = 1
inference_config.ipu.batches_per_step = 256
inference_config.ipu.gradient_accumulation = 16
inference_config.optimizer.learning_rate=.000025
inference_config.ipu.ipus_per_replica = 1

print("A", inference_config)

def one_hot(a, num_classes):
    data = np.zeros((num_classes,),dtype=np.float)
    for n in a:
        data[n] = 1
    return data

def create_config(train):
    from bert_model.modeling import handle_custom_ops

    train = train
    config = AutoConfig.from_pretrained('bert-base-uncased')
    config.training = train
    config.embedding_serialization_factor=1
    config.num_labels = 10
    if train:
        config.layers_per_ipu = [0,4,4,4]#self.inference_config.ipu.layers_per_ipu
        config.recompute_checkpoint_every_layer=True
        inference_config.detail.batch_size=2
        inference_config.detail.sequence_length=384

    config.recompute_checkpoint_every_layer=False
    config.problem_type = 'multi_label_classification'



    return config

config = create_config(True)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
dataset = load_dataset("xed_en_fi", "en_annotated")['train']
tokenized_dataset = dataset.map(lambda x: tokenizer(x['sentence'],
    max_length=inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True),batched=True)

tokenized_dataset = tokenized_dataset.map(lambda x: {'label':one_hot(x['labels'],config.num_labels)})
print("C", tokenized_dataset[0])

columns = ['input_ids', 'attention_mask','label']
tokenized_dataset.set_format(type='torch', columns=columns)
print("B", len(tokenized_dataset))

options = get_options(inference_config.ipu, True)

data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=inference_config.detail.batch_size, shuffle=True)

model = PipelinedBertForSequenceClassification.from_pretrained('bert-base-uncased',config=config).half()
optimizer = get_optimizer(inference_config.optimizer, model)
model.train()
model_ipu = poptorch.trainingModel(model, options, optimizer)

iter_loader = iter(data_loader)
print("A", len(iter_loader))
#data = next(iter_loader)
for x in range(10):
    try :
        pass
        data = next(iter_loader)
    except Exception as e:
        iter_loader = iter(data_loader)
        data = next(iter_loader)
    tic = time.time()

    output = model_ipu(data['input_ids'], data['attention_mask'], data['label'])
    print("B", output[0], inference_config.detail.batch_size*inference_config.ipu.batches_per_step*inference_config.ipu.gradient_accumulation/(time.time()-tic))
    print("C", output[1].shape)
model_ipu.save_pretrained("imdb_checkpoint")
