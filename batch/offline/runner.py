
from typing import Optional, List
from dataclasses import dataclass
from celery_worker import  run_infer
import dataclasses
import pymongo

from .offline_config import InferDescription
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

    def create_model_description(self):
        infer_description = InferDescription()
        infer_description.name = self.name
        infer_description.tokenizer = self.tokenizer
        infer_description.checkpoint = self.checkpoint
        infer_description.dataset = self.dataset
        infer_description.classifier.classifier_type = self.classifier
        infer_description.classifier.num_labels = self.num_labels
        
        return infer_description

@dataclass
class ModelResponse:
    request_id:str

def run(model_input:InferConfig) -> ModelResponse:

    model_description = model_input.create_model_description()
    mongo, result_id = create_mongo_interface(model_description)

    model_description_dict = dataclasses.asdict(model_description)
    uuid = run_infer.delay(model_description_dict, str(result_id))
    # Attach the ID to the Database
    mongo.update_id(result_id, str(uuid))
    print("Running Inference", result_id)
    return ModelResponse(str(result_id))

#def get_results():
#    mongo = MongoInterface()
#    results = mongo.get_all_results()
#    return results