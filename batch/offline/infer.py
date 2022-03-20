from transformers import AutoTokenizer, AutoConfig
from .ipu_options import get_options
from datasets import load_dataset

import poptorch
from celery import states
import dataclasses

from .offline_config import InferDescription
import os
import ctypes
import sys
import logging
from .bert_model.modeling import PipelinedBertForSequenceClassification, PipelinedBertForTokenClassification

logger = logging.getLogger()
import traceback
import numpy as np
import time

def create_data_loader(inference_config:InferDescription, tokenizer, options):
    dataset = load_dataset(inference_config.dataset)
    dataset = dataset['test']
    #print(dataset)

    tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
        max_length=inference_config.detail.sequence_length, truncation=True, padding=True),batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask','label'])
    data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=inference_config.detail.batch_size, shuffle=True)

    return data_loader

def handle_custom_ops(config):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
    if os.path.exists(CUSTOM_OP_PATH):
        ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
        ops_and_patterns.setVocabSize(config.vocab_size)
        ops_and_patterns.setEmbeddingSize(config.hidden_size)
        ops_and_patterns.setHiddenSize(config.hidden_size)
    else:
        exit()

def handle_error(message:str, e=None):
    logger.error(message)
    #if e is not None:
    #    traceback.print_exception(e)
    sys.exit(1)

def update_status(mongo, t, m=None):
    if mongo is not None:
        mongo.update_status(t,m)

def main(inference_config:InferDescription, mongo, celery, logger):
    
    update_status(mongo, "Data")
    try :
        options = get_options(inference_config.ipu)
        config = AutoConfig.from_pretrained(inference_config.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(inference_config.tokenizer, use_fast=True)
        data_loader = create_data_loader(inference_config, tokenizer, options)
    except Exception as e:
        update_status(mongo, "DataError",str(e))
        handle_error(f"Data Loading Error {str(e)}",e)
        return

    update_status(mongo, "Model")
    try :
        config = AutoConfig.from_pretrained(inference_config.checkpoint)
        config.embedding_serialization_factor=inference_config.ipu.embedding_serialization_factor
        config.num_labels = inference_config.classifier.num_labels
        config.layers_per_ipu = [24]
        config.recompute_checkpoint_every_layer=False
        
        if inference_config.classifier.classifier_type == 'Sequence':
            model = PipelinedBertForSequenceClassification.from_pretrained(inference_config.checkpoint, config=config).half()
        elif inference_config.classifier.classifier_type == 'Token':
            model = PipelinedBertForTokenClassification.from_pretrained(inference_config.checkpoint, config=config).half()
        elif inference_config.classifier.classifier_type == 'MLM':
            handle_error("MLM Current Not Supported")
        else:
            handle_error("Classifier Not Found")

        model_ipu = poptorch.inferenceModel(model, options)
    except Exception as e:
        update_status(mongo, "ModelError",str(e))
        logger.info(f"Model Compilation Error {str(e)}")
        return 

    update_status(mongo, "Running")

    iter_loader = iter(data_loader)
    results = []
    epoch = 0 
    step = 0

    errors,samples = 0,0
    while True:
        tic = time.time()
        try :
            data = next(iter_loader)
        except Exception as e:
            break


        result = model_ipu(data['input_ids'],
            data['attention_mask'],
            data['token_type_ids'])

        error = (result[1] - data['label']).numpy()
        errors += np.count_nonzero(error)
        samples += len(error)
        logger.info(f"Accuracy {errors/samples}")
        result = {
            'epoch':epoch,
            'time': time.time() - tic,
            'error':0.0,
            'accuracy':errors/samples
        }
        
        if mongo is not None:
            mongo.update_result(result)

        step += 1



        
    model_ipu.detachFromDevice()


    return {"status":"Success", "results":results}

if __name__ == "__main__":
    inference_config = InferDescription()
    main(inference_config)