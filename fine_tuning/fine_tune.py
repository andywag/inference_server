from transformers import AutoTokenizer, AutoConfig
from fine_tune_config import ModelDescription, ModelResult
from ipu_options import get_options
from datasets import load_dataset
from optimization import get_optimizer

import poptorch
from celery import states
import pymongo
import dataclasses
from bson.objectid import ObjectId
from mongo_interface import MongoInterface
import time
import socket

def create_data_loader(model_description:ModelDescription, tokenizer, options):
    dataset = load_dataset(model_description.dataset.name)
    if model_description.dataset.train is not None:
        dataset = dataset[model_description.dataset.train]
    tokenized_dataset = dataset.map(lambda x: tokenizer(x[model_description.dataset.text],
        max_length=model_description.model_specific.sequence_length, truncation=True, padding=True),batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'token_type_ids', 'attention_mask', 'label'])
    data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=model_description.execution_description.batch_size, shuffle=True)

    return data_loader



def main(model_description:ModelDescription, result_id:str, celery, logger):

    model_description = model_description.model_description
    mongo = MongoInterface()

    result_id = ObjectId(result_id)
    # Create Result In Mongo Database
    logger.info(f"Update Status {result_id}")
    mongo.update_status(result_id,"LoadingData")
    try :
        options = get_options(model_description)
        config = AutoConfig.from_pretrained(model_description.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(model_description.tokenizer, use_fast=True)
        data_loader = create_data_loader(model_description, tokenizer, options)
    except Exception as e:
        mongo.update_status(result_id,"Data Error",str(e))
        logger.info(f"Data Loading Error {str(e)}")

    try :
        mongo.update_status(result_id,"Compiling")
        model = model_description.get_model(config,half=True)
        optimizer = get_optimizer(model_description, model)
        model.train()
        model_ipu = poptorch.trainingModel(model, options, optimizer)
    except Exception as e:
        mongo.update_status(result_id,"Compile Error",str(e))
        logger.info(f"Model Compilation Error {str(e)}")

    mongo.update_status(result_id,"Running")

    iter_loader = iter(data_loader)
    results = []
    epoch = 0 
    step = 0
    while True:
        try :
            data = next(iter_loader)
        except :
            if model_description.execution_description.epochs is not None and epoch == model_description.execution_description.epochs:
                break
            iter_loader = iter(data_loader)
            data = next(iter_loader)  
            epoch += 1         
        
        tic = time.time()
        result = model_ipu(data['input_ids'],
            data['attention_mask'],
            data['token_type_ids'],
            data['label'])
        delta = time.time() - tic

        result_data = float(result[0][0].data)
        result = {
            'epoch':epoch,
            'time': delta,
            'error':result_data,
            'accuracy':0.0
        }

        mongo.update_result(result_id, result)

        logger.info(f"{epoch} - {step} : {result_data}")
        results.append(result_data)
        step += 1
        if model_description.execution_description.training_steps is not None and step == model_description.execution_description.training_steps:
            break
        
    model_ipu.detachFromDevice()

    mongo.update_status(result_id,"Finished")

    return {"status":"Success", "results":results}

