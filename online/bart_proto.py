
import sys
sys.path.append('bart')
from bart_interface import BartInterfaceWrapper

import numpy as np
import attr
from dataclasses import dataclass
from typing import List

from general_fastapi import BasicFastApi
from api_classes import Bart, BartResponse
from transformers import BartTokenizerFast
import time
from model_proto import ModelProto

@dataclass
class BartInput:
    input_ids:List[List[int]]
    attention_mask:List[List[int]]

    def items(self):
        return [np.asarray(self.input_ids), np.asarray(self.attention_mask)]

@dataclass
class BartOutput:
    result:List[List[int]]

    @staticmethod
    def create(data:np.ndarray):
        return BartOutput(data.tolist())

class BartApi(BasicFastApi):        
    def __init__(self, proto):
        super().__init__(proto, 'bart')
        self.input_type = Bart
        self.output_type = BartResponse
        self.tokenizer = BartTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    
    def create_rabbit_input(self, bart:Bart):
        result = self.tokenizer.encode_plus(bart.text, max_length=512, padding='max_length')
        input_ids = result['input_ids']
        attention_mask = result['attention_mask']

           
        return BartInput([input_ids], [attention_mask]), None

    def handle_rabbit_output(self, response, state, tic):
        logits = np.asarray(response["result"])
    
        results = self.tokenizer.decode(logits[0,2:])

        return BartResponse(results, time.time() - tic)

   
    def create_input(self, ner:Bart, triton_input):
        
        result = self.tokenizer.encode_plus(ner.text, max_length=512, padding='max_length',return_tensors="pt")
        input_ids = result['input_ids'].numpy()
        attention_mask = result['attention_mask'].numpy()

        print("Here", ner.text)
        #full_sum = np.sum(np.asarray(result['attention_mask'],dtype=np.uint32))
        input_ids = np.asarray([input_ids],dtype=np.int32)
        attention_mask = np.asarray([attention_mask],dtype=np.int32)

        triton_input[0].set_data_from_numpy(input_ids)
        triton_input[1].set_data_from_numpy(attention_mask)
        
        return None

    def handle_output(self, response, state, tic):
        #full_sum = state[0]
        #offset_mapping = state[1]
        data = response('result')
        logits = np.asarray(data)
        real_logits = logits.reshape([1,32])

        results = self.tokenizer.decode(real_logits[0,2:])

        return BartResponse(results, time.time() - tic)





@dataclass
@attr.s(auto_attribs=True)
class BartProto(ModelProto):
    name:str = "bart"
    checkpoint:str ='ainize/bart-base-cnn'
    input_type = BartInput
    output_type = BartOutput

    def create_model(self):
        base_path = "./bart"
        model = BartInterfaceWrapper()
        return model

    def get_fast_apis(self):
        return [BartApi(self)]