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
from .bert_model.modeling import PipelinedBertForSequenceClassification, PipelinedBertForTokenClassification, PipelinedBertForPretraining

logger = logging.getLogger()
import traceback
import numpy as np
import time
import torch

def create_data_loader(inference_config:InferDescription, tokenizer, options):
    infer_dataset = inference_config.dataset
    data_tag = infer_dataset.split(",")

    data_internal = data_tag[0].split(":")
    if len(data_internal) == 1:
        dataset = load_dataset(data_internal[0])
    else:
        print("Load", data_internal)
        dataset = load_dataset(data_internal[0], data_internal[1])
    print("Starting", data_tag)
    for tag in data_tag[1:-1]:
        dataset = dataset[tag]
   

    tokenized_dataset = dataset.map(lambda x: tokenizer(x[data_tag[-1]],
        max_length=inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True),batched=True)
    if inference_config.classifier.classifier_type == 'Sequence':
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask','label'])
    else:
        tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask'])
    
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
            config.pred_head_transform = False
            model = PipelinedBertForPretraining.from_pretrained(inference_config.checkpoint, config=config).half()
        else:
            handle_error("Classifier Not Found")

        model_ipu = poptorch.inferenceModel(model, options)
        

    except Exception as e:
        update_status(mongo, "ModelError",str(e))
        logger.info(f"Model Definition Error {str(e)}")
        return 

    iter_loader = iter(data_loader)
    update_status(mongo, "Compiling")
    tic = time.time()
    try :
        first_data = next(iter_loader)
        input_ids = torch.zeros(first_data['input_ids'].shape,dtype=torch.int32)
        model_ipu.compile(input_ids,input_ids,input_ids)
    except Exception as e:
        update_status(mongo, "CompileError",str(e))
        logger.info(f"Model Definition Error {str(e)}")
        return 

    update_status(mongo, "Running")

    
    results = []
    epoch = 0 
    step = 0

    errors,samples = 0,0
    
    start_time = time.time()
    while True:
        
        if first_data is not None:
            data = first_data
            first_data = None
        else:
            try :
                data = next(iter_loader)
            except Exception as e:
                logger.info(f"Finished with Dataset {e}")
                break

        try :
            #tic = time.time()

            result = model_ipu(data['input_ids'],
                data['attention_mask'],
                data['token_type_ids'])
            
        except Exception as e:
            update_status(mongo, "RunError",str(e))
            logger.info(f"Finished with Dataset {e}")
            return

        samples += len(result[1])
        if 'label' in data:
            error = (result[1] - data['label']).numpy()
            errors += np.count_nonzero(error)
            logger.info(f"Accuracy {errors/samples} QPS {samples/(time.time()-start_time)}")
       
            if mongo is not None:
                mongo.update_accuracy(accuracy=1.0-errors/samples,qps=samples/(time.time()-start_time))
        else:
            
            logger.info(f"QPS {samples/(time.time()-start_time)}")
            mongo.update_accuracy(accuracy=0.0,qps=samples/(time.time()-start_time))
        step += 1
    update_status(mongo, "Finished")

        



        
    model_ipu.detachFromDevice()


    return {"status":"Success", "results":results}

if __name__ == "__main__":
    inference_config = InferDescription()
    main(inference_config)