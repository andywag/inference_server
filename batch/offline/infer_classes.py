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
import json

class Base:
    def __init__(self, inference_config:InferDescription):
        self.inference_config = inference_config
        self.dataset_columns = ['input_ids', 'token_type_ids', 'attention_mask']
        self.result_store = []
        self.label_store = []
        
    def create_config(self):
        config = AutoConfig.from_pretrained(self.inference_config.checkpoint)
        config.embedding_serialization_factor=self.inference_config.ipu.embedding_serialization_factor
        config.num_labels = self.inference_config.classifier.num_labels
        config.layers_per_ipu = [24]
        config.recompute_checkpoint_every_layer=False
        return config

    def compile_inputs(self, first_data):
        input_ids = torch.zeros(first_data['input_ids'].shape,dtype=torch.int32)
        return [input_ids, input_ids, input_ids]

    def model_inputs(self, data):
        return [data['input_ids'], data['token_type_ids'], data['attention_mask']]

    def handle_result(self, result, data):
        self.result_store.append((result))
        if 'label' in data:
            self.label_store.append(data['label'])
          
        return None

    def post_process(self):
        pass


class Sequence(Base):
    def __init__(self, inference_config):
        super().__init__(inference_config)
        self.model = PipelinedBertForSequenceClassification

    def handle_result(self, result, data):
        #super().handle_result(result,data)
        super().handle_result(result,data)
        if 'label' in data:
            for x in range(len(result[0])):
                index = result[1][x].item()
                #self.result_store.append((index, result[0][x][index]))
            error = (result[1] - data['label']).numpy()
            error = np.count_nonzero(error)
            return error
        else:
            return None

    def post_process(self):
        results = []
        import pickle
        pickle.dump(self.result_store, open( "save.pik", "wb" ) )
        count = 0
        for x in range(len(self.result_store)):
            result = self.result_store[x]
            for y in range(len(result[1])):
                count += 1
                #logger.info(f"B {result[1]} {result[1].shape} {y} {result[1][y]}")
                index = result[1][y].item()
                #print("A", result[0].shape, result[0][y][index].item())
                results.append({'class':index,'probability':result[0][y][index].item()})

        print("Dumping Resutls", count, len(results))
        with open('result.json', 'w') as outfile:
            json.dump(results, outfile)
            

class Token(Base):
    def __init__(self, inference_config):
        super().__init__(inference_config)
        self.model = PipelinedBertForTokenClassification

        

class MLM(Base):
    def __init__(self, inference_config):
        super().__init__(inference_config)
        self.model = PipelinedBertForPretraining

    def create_config(self):
        config = super().create_config()
        config.pred_head_transform = False
        return config

    def compile_inputs(self, first_data):
        inputs = super().compile_inputs(first_data)
        
        input_shape = first_data['input_ids'].shape
        input_ids = torch.zeros(first_data['input_ids'].shape,dtype=torch.int32)
        position_shape = list(input_shape)
        position_shape[1] = self.inference_config.classifier.num_labels
        position_mask = torch.zeros(position_shape, dtype=torch.int64)
        self.position_mask = position_mask
        inputs.append(position_mask)
        return inputs

    def model_inputs(self, data):
        m_inputs = super().model_inputs(data)
        m_inputs.append(self.position_mask)
        return m_inputs
