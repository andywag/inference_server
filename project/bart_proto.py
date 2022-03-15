
import sys
sys.path.append('bart')
from bart_interface import BartInterfaceWrapper

import numpy as np
import attr
from dataclasses import dataclass
from model_proto import ModelProto, SignalProto
from typing import List

from general_fastapi import BasicFastApi
from api_classes import Bart, BartResponse
from transformers import BartTokenizerFast
import time

input_ids = SignalProto('input_ids', np.zeros((1,512),dtype=np.int32))
attention_mask = SignalProto('attention_mask', np.zeros((1,512),dtype=np.int32))
result_ids = SignalProto('result_ids', np.zeros((1,32),dtype=np.int32))

class BartApi(BasicFastApi):        
    def __init__(self, proto):
        super().__init__(proto, 'bart')
        self.input_type = Bart
        self.output_type = BartResponse
        self.tokenizer = BartTokenizerFast.from_pretrained("ainize/bart-base-cnn")
    
    def create_input(self, ner:Bart, triton_input):
        #result = self.tokenizer.encode_plus(text = ner.text, max_length=512, 
        #    padding='max_length',return_offsets_mapping=True)
        
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
        
        logits = response.as_numpy("result_ids")
        real_logits = logits.reshape([1,32])

        results = self.tokenizer.decode(real_logits[0,2:])

        return BartResponse(results, time.time() - tic)





@dataclass
@attr.s(auto_attribs=True)
class BartProto(ModelProto):
    name:str = "bart"
    max_batch_size:int = 512
    checkpoint:str ='ainize/bart-base-cnn'
    inputs:List[SignalProto] = [input_ids, attention_mask]
    outputs:List[SignalProto] = [result_ids]

    def create_model(self):
        base_path = "./bart"
        model = BartInterfaceWrapper()
        return model

    def get_fast_apis(self):
        return [BartApi(self)]