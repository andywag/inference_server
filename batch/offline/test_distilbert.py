from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch
import poptorch
from ipu_options import get_options
from optimization import get_optimizer

from offline_config import InferDescription
from bert_model.modeling import PipelinedDistilBertForSequenceClassification
import time

inference_config = InferDescription()
inference_config.detail.batch_size = 1
inference_config.ipu.batches_per_step = 128
inference_config.ipu.gradient_accumulation = 16
inference_config.optimizer.learning_rate=.0001
inference_config.ipu.ipus_per_replica = 1

print("A", inference_config)

#checkpoint = 'textattack/distilbert-base-uncased-imdb'
checkpoint = 'distilbert-base-uncased'

def create_config(train):
    from bert_model.modeling import handle_custom_ops
    train = train
    config = AutoConfig.from_pretrained(checkpoint)
    config.training = train
    config.embedding_serialization_factor=1
    config.num_labels = 10
    if train:
        config.layers_per_ipu = [0,4,4,4]#self.inference_config.ipu.layers_per_ipu
        config.recompute_checkpoint_every_layer=True
        inference_config.detail.batch_size=32
        inference_config.detail.sequence_length=384

    config.recompute_checkpoint_every_layer=False
    config.problem_type = 'multi_label_classification'


    return config


tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)
dataset = load_dataset('imdb')['train']
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
    max_length=inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True),batched=True)

columns = ['input_ids', 'attention_mask','label']
tokenized_dataset.set_format(type='torch', columns=columns)
options = get_options(inference_config.ipu, True)

data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=1, shuffle=True)

config = create_config(True)
model = PipelinedDistilBertForSequenceClassification.from_pretrained(checkpoint,config=config).half()
optimizer = get_optimizer(inference_config.optimizer, model)
model.train()
model_ipu = poptorch.trainingModel(model, options, optimizer)

iter_loader = iter(data_loader)

print("Here")
data = next(iter_loader)
for x in range(100):
    try :
        pass
        data = next(iter_loader)
    except Exception as e:
        iter_loader = iter(data_loader)
        data = next(iter_loader)
    tic = time.time()
    #torch.set_printoptions(profile="full")
    #print("R",  data['label'].shape, data['input_ids'].shape)
    #torch.set_printoptions(profile="default") # reset

    output = model_ipu(data['input_ids'], data['attention_mask'], data['label'])
    print("B", output[0], inference_config.ipu.batches_per_step*inference_config.ipu.gradient_accumulation/(time.time()-tic))
    #print("C", output)
