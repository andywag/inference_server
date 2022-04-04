
from typing import Optional, List
from dataclasses import dataclass
from celery_worker import  run_infer
import dataclasses
import pymongo

from offline.offline_config import InferDescription
from mongo_common import create_mongo_interface

@dataclass
class InferConfig:
    name:str
    model_type:str
    checkpoint:str
    tokenizer:str
    dataset:str
    classifier:str
    num_labels:int
    cloud:str
    endpoint:str
    result_folder:str
    # Training Only
    optimizer:str=""
    learning_rate:float=0.0001
    epochs:int=1

    def create_model_description(self):
        infer_description = InferDescription()
        infer_description.name = self.name
        infer_description.tokenizer = self.tokenizer
        infer_description.checkpoint = self.checkpoint
        infer_description.dataset = self.dataset
        infer_description.classifier.classifier_type = self.classifier
        infer_description.classifier.num_labels = self.num_labels
        infer_description.cloud = self.cloud
        infer_description.endpoint = self.endpoint
        infer_description.result_folder = self.result_folder
        if self.classifier == 'MLM':
            infer_description.detail.batch_size = 8

        # Training Only
        infer_description.optimizer.epochs = self.epochs
        infer_description.optimizer.learning_rate = self.learning_rate
        
        return infer_description

@dataclass
class ModelResponse:
    request_id:str



def run(model_input:InferConfig, train:bool=False) -> ModelResponse:
    model_description = model_input.create_model_description()
    if train:
        model_description.train = True
    mongo, result_id = create_mongo_interface(model_description)

    model_description_dict = dataclasses.asdict(model_description)
    uuid = run_infer.delay(model_description_dict, str(result_id))
    # Attach the ID to the Database
    mongo.update_id(str(uuid))
    mongo.update_status("Submitted")
    print("Running Inference", result_id, train)
    return ModelResponse(str(result_id))

