from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset
import torch
import poptorch
from ipu_options import get_options
from optimization import get_optimizer

from offline_config import InferDescription
from bert_model.modeling import PipelinedBertForSequenceClassification
import time

inference_config = InferDescription()
inference_config.detail.batch_size = 20
inference_config.ipu.batches_per_step =64

print("A", inference_config)

checkpoint = 'imdb_checkpoint'

def create_config(train):
    from bert_model.modeling import handle_custom_ops

    train = train
    config = AutoConfig.from_pretrained(checkpoint)
    config.embedding_serialization_factor=1
    config.num_labels = 10
    config.layers_per_ipu = [24]    
    config.recompute_checkpoint_every_layer=False

    handle_custom_ops(config)
    config.problem_type = 'multi_label_classification'


    return config


tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased', use_fast=True)
dataset = load_dataset('imdb')['test']
tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
    max_length=inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True),batched=True)

columns = ['input_ids', 'attention_mask','label']
tokenized_dataset.set_format(type='torch', columns=columns)

options = get_options(inference_config.ipu, False)

data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=inference_config.detail.batch_size, shuffle=True)

config = create_config(True)
model = PipelinedBertForSequenceClassification.from_pretrained(checkpoint,config=config).half()
model_ipu = poptorch.inferenceModel(model, options)

iter_loader = iter(data_loader)

#data = next(iter_loader)
for x in range(100):
    try :
        pass
        data = next(iter_loader)
    except Exception as e:
        break
        #iter_loader = iter(data_loader)
        #data = next(iter_loader)
    tic = time.time()

    output = model_ipu(data['input_ids'], data['attention_mask'], data['label'])
    error = torch.sum(torch.abs(data['label'] - output[1]))
    print("B", (len(data['label'])-error)/len(data['label']), inference_config.detail.batch_size*inference_config.ipu.batches_per_step/(time.time()-tic))
    #print("E", error)