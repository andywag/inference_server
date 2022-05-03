from typing import Optional, List
from dataclasses import dataclass
import dataclasses

from offline.offline_config import InferDescription

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
        infer_description.model_type = self.model_type
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
