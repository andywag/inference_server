from transformers import AutoTokenizer, AutoConfig
from fine_tune_config import ModelDescription, ModelResult
from ipu_options import get_options
from datasets import load_dataset
from optimization import get_optimizer

import poptorch
import pymongo
import dataclasses
from bson.objectid import ObjectId
from mongo_interface import MongoInterface
import time
import socket

import logging 
logger = logging.getLogger('test')

def create_data_loader(model_description:ModelDescription, model_specific, tokenizer, options):
    dataset = load_dataset(model_description.dataset.name)
    if model_description.dataset.train is not None:
        dataset = dataset[model_description.dataset.train]
    tokenized_dataset = dataset.map(lambda x: tokenizer(x[model_description.dataset.text],
        max_length=model_specific.sequence_length, truncation=True, padding=True),batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=model_description.execution_description.batch_size, shuffle=True)

    return data_loader



def main(top_description):

    #model_description = model_description.model_description
    model_specific = top_description.model_specific
    model_description = top_description.model_description

    mongo = MongoInterface()

    # Create Result In Mongo Database
    try :
        options = get_options(model_description)
        config = AutoConfig.from_pretrained(model_description.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_description.tokenizer, use_fast=True)
        data_loader = create_data_loader(model_description, model_specific, tokenizer, options)
    except Exception as e:
        logger.info(f"Data Loading Error {str(e)}")
        return

    try :
        model = top_description.get_model(config, logger, half=True)
        optimizer = get_optimizer(model_description, model)
        model.train()
        model_ipu = poptorch.trainingModel(model, options, optimizer)
    except Exception as e:
        logger.info(f"Model Definition Error {str(e)}")
        return 

    iter_loader = iter(data_loader)

    # Compile IPU Model
    first_data = True
    try :
        data = next(iter_loader)
        
        #model_ipu.compile(data['input_ids'],
        #    data['attention_mask'],
        #    data['token_type_ids'],
        #    data['label'])

    except Exception as e:
        logger.info(f"Compilation Error {str(e)}")
        return 


    
    start_time = time.time()

    results = []
    epoch = 0 
    step = 0
    samples = 0
    while True:
        if not first_data:
            try :
                data = next(iter_loader)
            except :
                if model_description.execution_description.epochs is not None and epoch == model_description.execution_description.epochs-1:
                    break
                iter_loader = iter(data_loader)
                data = next(iter_loader)  
                epoch += 1
        else:
            first_data = False         
        
        tic = time.time()
        result = model_ipu(data['input_ids'],
            data['attention_mask'],
            data['token_type_ids'],
            data['label'])

        #logger.info("A", model_description)
        samples += model_description.execution_description.batch_size*model_description.execution_description.batches_per_step*model_description.execution_description.gradient_accumulation 
        result_data = float(result[0][0].data)
        delta = time.time() - tic

        
        result = {
            'epoch':epoch,
            'time': delta,
            'error':result_data,
            'accuracy':0.0
        }


        logger.info(f"{epoch} - {step} - {samples}: {result_data}")
        results.append(result_data)
        step += 1
        if model_description.execution_description.training_steps is not None and step == model_description.execution_description.training_steps:
            break
        
    model_ipu.detachFromDevice()


    return {"status":"Success", "results":results}

from bert_model.bert_config_new import BertDescription
if __name__ == '__main__':
    print("Hello World")
    description = BertDescription()
    main(description)

