
from bert_proto import *
import sys
sys.path.append("./gpt2_model")
from gpt2_interface import GPT2TritonWrapper

import numpy as np
import attr
from dataclasses import dataclass
from model_proto import ModelProto, SignalProto
from typing import List
from bert_proto import *

input_ids = SignalProto('input_ids',np.zeros(shape=128, dtype=np.uint32))
input_length = SignalProto('input_length',np.zeros(shape=1, dtype=np.uint32))
output_length = SignalProto('output_length',np.zeros(shape=1, dtype=np.uint32))

token_ids = SignalProto('token_ids',np.zeros(shape=640, dtype=np.uint32))

from general_fastapi import BasicFastApi
from api_classes import GPT2, GPT2Response
from transformers import GPT2TokenizerFast
import torch

import time


class GPT2Api(BasicFastApi):        
    def __init__(self, proto):
        super().__init__(proto, 'gpt')
        self.input_type = GPT2
        self.output_type = GPT2Response
        self.tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    def create_rabbit_input(self, gpt:GPT2):
        text_ids = self.tokenizer.encode(gpt.text, add_special_tokens=False)
        input_ids = np.zeros((1, 128), dtype=np.uint32)
        input_length = np.zeros((1, 1), dtype=np.uint32)
        output_length = np.zeros((1, 1), dtype=np.uint32)
        input_ids[:,:len(text_ids)] = text_ids
        input_length[:,0] = len(text_ids)
        output_length[:,0] = 128

        return Gpt2Input(input_ids.tolist(), input_length.tolist(), output_length.tolist()), None

    def handle_rabbit_output(self, response, state, tic):
        logits = np.asarray(response['result'])
        real_logits = logits[0][logits[0] != 0]
        text = self.tokenizer.decode(real_logits.tolist())
        return GPT2Response(text, time.time() - tic)

            
    def create_input(self, gpt:GPT2, triton_input):
        text_ids = self.tokenizer.encode(gpt.text, add_special_tokens=False)
        #print("A", text_ids)
        input_ids = np.zeros((1, 128), dtype=np.uint32)
        input_length = np.zeros((1, 1), dtype=np.uint32)
        output_length = np.zeros((1, 1), dtype=np.uint32)
        input_ids[:,:len(text_ids)] = text_ids
        input_length[:,0] = len(text_ids)
        output_length[:,0] = 128

        triton_input[0].set_data_from_numpy(input_ids)
        triton_input[1].set_data_from_numpy(input_length)
        triton_input[2].set_data_from_numpy(output_length)
        
        return None

    def handle_output(self, response, state, tic):
        logits = response.as_numpy("token_ids")
        real_logits = logits[0][logits[0] != 0]
        
        text = self.tokenizer.decode(real_logits.tolist())
        
        return GPT2Response(text, time.time() - tic)


@dataclass 
class Gpt2Input:
    input_ids:List[List[int]]
    input_length:List[List[int]]
    output_length:List[List[int]]

    def items(self):
        return [self.input_ids, self.input_length[0], self.output_length[0]]

@dataclass
class Gpt2Output:
    result:List[List[int]]

    @staticmethod
    def create(data:np.ndarray):
        return Gpt2Output(data.tolist())


@dataclass
@attr.s(auto_attribs=True)
class GPT2Proto(ModelProto):
    name:str = "gpt2"
    max_batch_size:int = 512
    checkpoint:str ='https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin'
    inputs:List[SignalProto] = [input_ids, input_length, output_length]
    outputs:List[SignalProto] = [token_ids]

    input_type = Gpt2Input
    output_type = Gpt2Output

    def create_model(self):
        model = GPT2TritonWrapper() 
        return model

    def get_fast_apis(self):
        return [GPT2Api(self)]
    




