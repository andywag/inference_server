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
import pickle

class Base:
    def __init__(self, inference_config:InferDescription):
        self.inference_config = inference_config
        self.dataset_columns = ['input_ids', 'token_type_ids', 'attention_mask']
        self.result_store = []
        self.label_store = []

    def tokenize(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
            max_length=self.inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True),batched=True)
        return tokenized_dataset

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

    def post_process(self, mongo, cloud_file_system=None):
        pass

    def output_results(self, mongo, total_results, cloud_file_system=None):
        mongo.put_result(total_results)

        if self.inference_config.result is not None and cloud_file_system is not None:
            result_file = self.inference_config.result.replace("cloud:","")
            with fs.open(result_file, 'wb') as fp:
                json.dump(total_results, fp)


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

    def post_process(self, mongo, cloud_file_system=None):
        results = []
        import pickle
        count = 0
        for x in range(len(self.result_store)):
            result = self.result_store[x]
            for y in range(len(result[1])):
                count += 1
                index = result[1][y].item()
                results.append({'class':index,'probability':result[0][y][index].item()})
        
        self.output_results(mongo, results, cloud_file_system)
        
        
            

class Token(Base):
    def __init__(self, inference_config):
        super().__init__(inference_config)
        self.model = PipelinedBertForTokenClassification
        self.offset_store = []

    def tokenize(self, tokenizer, dataset):
        tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
            max_length=self.inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True,return_offsets_mapping=True,return_length=True),batched=True)
        self.offset_store=tokenized_dataset['offset_mapping']
        self.length_store=tokenized_dataset['length']
        return tokenized_dataset

    def handle_result(self, result, data):
        self.result_store.append((result))
        if 'label' in data:
            self.label_store.append(data['label'])
        
        return None

    def post_process(self, mongo, cloud_file_system=None):
        data = self.result_store
        offsets = self.offset_store
        offset_index = 0
        results = []
        for x in range(len(data)):
            for z in range(len(data[x][0])):
                probabilities = data[x][0][z]
                indices = data[x][1][z]
                real_offset = offsets[offset_index]
                element_result = []
                for y in range(self.length_store[offset_index]):
                    if indices[y] != 0:
                        element = {'index':indices[y].item(),'logit':probabilities[y][indices[y]].item(),'range':offsets[offset_index][y]}
                        element_result.append(element)
                offset_index += 1
                results.append(element_result)
        self.output_results(mongo, results, cloud_file_system)
        

        

class MLM(Base):
    def __init__(self, inference_config):
        super().__init__(inference_config)
        self.model = PipelinedBertForPretraining
        self.masked_store = []

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
        
        position_values = (data['input_ids'] == 103)
        positions = position_values.nonzero()

        masked_lm_positions=torch.zeros(size=(len(data['input_ids']),self.inference_config.classifier.num_labels),dtype=torch.int)
        self.masked_store.append(masked_lm_positions)

        row_index = [0]*len(data['input_ids'])
        for x in range(len(positions)):
            row = positions[x][0]
            col = row_index[row]
            row_index[row] += 1
            value = positions[x][1]
            if row_index[row] < 32:
                masked_lm_positions[row][col] = value

        m_inputs.append(masked_lm_positions)
        return m_inputs

    def handle_result(self, result, data):
        self.result_store.append((result))
        if 'label' in data:
            self.label_store.append(data['label'])
        
        return None

    def post_process(self, mongo, cloud_file_system=None):
        data = self.result_store

        index = 0
        total_results = []
        for x in range(len(data[0])): # Batch Index
            for y in range(len(data[x][0][2])): # Item Index
                sequence_result = []
                for z in range(len(data[x][0][y])): # 
                    if self.masked_store[x][y][z] != 0:
                        tokens = self.tokenizer.convert_ids_to_tokens(data[x][1][y][z])
                        logits = [float(x) for x in data[x][0][y][z].numpy()]
                        result = {'index':self.masked_store[x][y][z].item(),'tokens':tokens, 'logits':logits}
                        sequence_result.append(result)
                total_results.append(sequence_result)

        self.output_results(mongo, total_results, cloud_file_system)


        #masked_lm_positions = self.masked_lm_positions
        #with open('save.pik','wb') as fp:
        #    pickle.dump({'data':data,'masked_lm_positions':self.masked_store},fp)

