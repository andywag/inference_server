
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dataclasses import dataclass
from .celery_worker import  run_dict
import dataclasses
from .fine_tune_config import ModelDescription, ModelResult
from .bert_model.bert_config import BertSpecific
import pymongo
from .mongo_interface import MongoInterface
from .bert_model.bert_config_new import BertDescription

@dataclass
class ModelConfig:
    name:str
    model_type:str
    checkpoint:str
    dataset:str
    tokenizer:str
    optimizer:str
    learning_rate:float
    epochs:int
    classifier:str
    num_labels:int

    def create_model_description(self, model_description=ModelDescription()):
        model_description.name = self.name
        model_description.tokenizer = self.tokenizer
        model_description.checkpoint = self.checkpoint
        model_description.execution_description.epochs = self.epochs
        model_description.execution_description.learning_rate = self.learning_rate
        model_description.dataset.name = self.dataset
        return model_description

@dataclass
class ModelResponse:
    request_id:str

def run(model_input:ModelConfig) -> ModelResponse:

    model_description = model_input.create_model_description()

    if model_input.model_type == 'BERT':
        bert_description = BertDescription()
        bert_description.model_description = model_description
        bert_description.model_specific.tuning_type = model_input.classifier
        bert_description.model_specific.num_labels = model_input.num_labels
        model_description = bert_description
        #model_description = model_input.create_model_description()
    else:
        mongo.update_status(result_id,"ModelNotSupported")
        print("Model Not Supported")

    mongo = MongoInterface()
    result = ModelResult("", "", model_description, list(), list())
    result_dict = dataclasses.asdict(result)
    result_id = mongo.create_result(result_dict)
    mongo.update_status(result_id,"Submit")
    
    print(model_input)
    model_description_dict = dataclasses.asdict(model_description)
    uuid = run_dict.delay(model_description_dict, str(result_id))
    # Attach the ID to the Database
    mongo.update_id(result_id, str(uuid))

    return ModelResponse(str(result_id))

def get_results():
    mongo = MongoInterface()
    results = mongo.get_all_results()
    return results