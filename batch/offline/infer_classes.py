from transformers import AutoTokenizer, AutoConfig
from datasets import load_dataset

from celery import states
import dataclasses

from .offline_config import InferDescription
import os
import ctypes
import sys
import logging

logger = logging.getLogger()
import traceback
import numpy as np
import time
import json
import pickle
import torch



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

    def create_config(self, train):
        from .bert_model.modeling import handle_custom_ops

        self.train = train
        config = AutoConfig.from_pretrained(self.inference_config.checkpoint)
        config.training = train
        config.embedding_serialization_factor=self.inference_config.ipu.embedding_serialization_factor
        config.num_labels = self.inference_config.classifier.num_labels
        if train:
            config.layers_per_ipu = self.inference_config.ipu.layers_per_ipu
            config.recompute_checkpoint_every_layer=True
            self.inference_config.detail.batch_size=4
        else:
            config.layers_per_ipu = [24]
        config.recompute_checkpoint_every_layer=False
        #config.problem_type = "single_label_classification"
        handle_custom_ops(config)


        return config

    #def compile_inputs(self, first_data):
    #    input_ids = torch.zeros(first_data['input_ids'].shape,dtype=torch.int32)
    #    return [input_ids, input_ids, input_ids]

    def model_inputs(self, data):
        return [data['input_ids'],  data['attention_mask'], data['token_type_ids']]

    def handle_result(self, result, data):
        self.result_store.append((result))
        if 'label' in data:
            self.label_store.append(data['label'])
          
        return None

    def post_process(self, mongo, cloud_file_system=None):
        pass

    def output_results(self, mongo, total_results, fs=None):
        mongo.put_result(total_results)

        logger.info(f"Writing Results {self.inference_config.result_folder}")

        if self.inference_config.result_folder is not None and self.inference_config.result_folder != "":
            fs.output_results(self.inference_config.result_folder, total_results)

class Sequence(Base):
    def __init__(self, inference_config):
        from .bert_model.modeling import PipelinedBertForSequenceClassification

        super().__init__(inference_config)
        self.model = PipelinedBertForSequenceClassification


    def model_inputs(self, data):
        if not self.train:
            return [data['input_ids'], data['attention_mask'], data['token_type_ids']]
        else:
            return [data['input_ids'], data['attention_mask'], data['token_type_ids'],  data['label']]

    def handle_result(self, result, data):
        #super().handle_result(result,data)
        super().handle_result(result,data)
        if self.train:
            logger.info("Training Results")
        else:
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
        if self.train:
            with open('save.pik','wb') as fp:
                pickle.dump({'result':self.result_store,'label':self.label_store}, fp)
        else:
            for x in range(len(self.result_store)):
                result = self.result_store[x]
                for y in range(len(result[1])):
                    count += 1
                    index = result[1][y].item()
                    results.append({'class':index,'probability':result[0][y][index].item()})
        
            self.output_results(mongo, results, cloud_file_system)
        
        
            

class Token(Base):
    def __init__(self, inference_config):
        from .bert_model.modeling import PipelinedBertForTokenClassification

        super().__init__(inference_config)
        self.model = PipelinedBertForTokenClassification
        self.offset_store = []

    def model_inputs(self, data):
        if not self.train:
            return [data['input_ids'], data['attention_mask'], data['token_type_ids']]
        else:
            return [data['input_ids'], data['attention_mask'], data['token_type_ids'],  data['label']]

    def tokenize(self, tokenizer, dataset):
        if self.train:
            def create_labels(data, mlen):
                new_results = [0]*mlen
                new_results[1:len(data)+1] = data
                return new_results
   
            def convert(data):
                return [" ".join(x) for x in data]
                #return " ".join(x)
            tokenized_dataset = dataset.map(lambda x: tokenizer(convert(x['tokens']),
                max_length=self.inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True,return_offsets_mapping=True,return_length=True),batched=True)
            
            tokenized_dataset = tokenized_dataset.map(lambda x: {'label':create_labels(x['tags'],self.inference_config.detail.sequence_length)})

        else:
            print("Base", dataset.column_names)
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
        if self.train:
            return 
            
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
        from .bert_model.modeling import PipelinedBertForPretraining

        self.model = PipelinedBertForPretraining
        self.masked_store = []

    def tokenize(self, tokenizer, dataset):
        self.tokenizer = tokenizer

        def create_position(input_ids, max_len):
            new_result = np.zeros(shape=(max_len,), dtype=np.int)
            result = np.where(np.asarray(input_ids) == 103)
    
            max_padding = min(max_len, len(result[0]))
            new_result[:max_padding] = result[0][:max_padding]
            return new_result

        sequence_length = self.inference_config.detail.sequence_length
        mask_length = self.inference_config.classifier.num_labels

        def create_labels(labels, mlen=32):
            def internal(x):
                new_result =[0]*mlen 
                result = tokenizer.convert_tokens_to_ids(x)
                new_len = min(mlen, len(result))
                new_result[:new_len] = result[:new_len]
                return new_result

            return [internal(x) for x in labels]

        #tokenized_dataset = dataset.map(lambda x: tokenizer(x['label_text'],
        #    max_length=mask_length, truncation=True, pad_to_max_length=True),batched=True)
        #tokenized_dataset = tokenized_dataset.map(lambda batch: {"masked_lm_labels": convert_labels(batch["input_ids"])}, batched=True)
        tokenized_dataset = dataset.map(lambda batch: {"masked_lm_labels": create_labels(batch["label_text"], mask_length)}, batched=True)


        tokenized_dataset = tokenized_dataset.map(lambda x: tokenizer(x['text'],
            max_length=sequence_length, truncation=True, pad_to_max_length=True),batched=True)

        tokenized_dataset = tokenized_dataset.map(lambda x: {'masked_lm_positions':create_position(x['input_ids'], mask_length)})

        print("B", tokenized_dataset.column_names)
        return tokenized_dataset

    def create_config(self, train:bool=False):
        config = super().create_config(train)
        config.pred_head_transform = False
        return config

    def compile_inputs(self, first_data):
        inputs = super().compile_inputs(first_data)
        
        input_shape = first_data['input_ids'].shape
        input_ids = torch.zeros(first_data['input_ids'].shape,dtype=torch.int32)
        position_shape = list(input_shape)
        position_shape[1] = self.inference_config.classifier.num_labels

        # Position Ids 
        position_mask = torch.zeros(position_shape, dtype=torch.int64)
        self.position_mask = position_mask
        inputs.append(position_mask)

        # Position Labels
        position_label = torch.zeros(position_shape, dtype=torch.int)
        inputs.append(position_label)
        
        return inputs

    def model_inputs(self, data):
        m_inputs = super().model_inputs(data)
        
        position_values = (data['input_ids'] == 103)
        positions = position_values.nonzero()

        masked_lm_positions=torch.zeros(size=(len(data['input_ids']),self.inference_config.classifier.num_labels),dtype=torch.int64)
        self.masked_store.append(masked_lm_positions)

        row_index = [0]*len(data['input_ids'])
        for x in range(len(positions)):
            row = positions[x][0]
            col = row_index[row]
            row_index[row] += 1
            value = positions[x][1]
            if row_index[row] < 32:
                masked_lm_positions[row][col] = value


        m_inputs.append(data['masked_lm_positions'])
        m_inputs.append(data['masked_lm_labels'])
        return m_inputs

    def handle_result(self, result, data):
        self.result_store.append((result[0]))
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


    

