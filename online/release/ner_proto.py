
#from bert_interface_wrapper import BertInterfaceWrapper

import sys
sys.path.append("../model_proto")
sys.path.append("../public_api")

import numpy as np
import attr
from dataclasses import dataclass
from model_proto import ModelProto
from typing import List
from bert_proto import *

from general_fastapi import BasicFastApi
from api_classes import Ner, NerResult, NerResponse
from transformers import BertTokenizerFast
import time

class NerApi(BasicFastApi):        
    def __init__(self, proto, host):
        super().__init__(proto, host, 'ner')
        self.input_type = Ner
        self.output_type = NerResponse
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
    
    def create_rabbit_input(self, ner:Ner):
        result = self.tokenizer.encode_plus(text = ner.text, max_length=384, 
            padding='max_length',return_offsets_mapping=True)
        full_sum = np.sum(np.asarray(result['attention_mask'],dtype=np.uint32))
        input_ids = np.asarray([result['input_ids']],dtype=np.uint32)
        token_type_ids = np.asarray([[full_sum]],dtype=np.uint32)
        offset_mapping = result['offset_mapping']

        return BertInput(input_ids.tolist(), token_type_ids.tolist(), [0]), (full_sum, offset_mapping)

    def handle_rabbit_output(self, response, state, tic:int):
        full_sum = state[0]
        offset_mapping = state[1]
        
        logits = np.asarray(response['logits'])
        real_logits = logits.reshape([384,9])

        results = []
        for x in range(full_sum):
            if real_logits[x][0] == -1.0:
                break
            logit = np.argmax(real_logits[x,:])
            if logit > 0:
                logit = np.argmax(real_logits[x,:])
                if real_logits[x,logit] > 2.0:
                    result = NerResult(offset_mapping[x][0],offset_mapping[x][1],int(logit),float(real_logits[x,logit]))
                    results.append(result)
        return NerResponse(results, time.time() - tic)



    def handle_output(self, response, state, tic):
        full_sum = state[0]
        offset_mapping = state[1]
        
        logits = response.as_numpy("logits")
        real_logits = logits.reshape([384,9])

        results = []
        for x in range(full_sum):
            if real_logits[x][0] == -1.0:
                break
            logit = np.argmax(real_logits[x,:])
            if logit > 0:
                logit = np.argmax(real_logits[x,:])
                if real_logits[x,logit] > 2.0:
                    result = NerResult(offset_mapping[x][0],offset_mapping[x][1],int(logit),float(real_logits[x,logit]))
                    results.append(result)
        return NerResponse(results, time.time() - tic)



@dataclass
@attr.s(auto_attribs=True)
class NerProto(ModelProto):
    name:str = "ner"
    checkpoint:str ='https://huggingface.co/dslim/bert-large-NER/resolve/main/pytorch_model.bin'

    input_type = BertInput
    output_type = NerOutput

    def create_model(self):
        from bert_interface_wrapper import BertInterfaceWrapper

        base_path = "./bert"
        config = f"{base_path}/configs/sut_inference_pack_384_ner_single.json"
        model = BertInterfaceWrapper( 
            config=config,
            ner=True)
        return model

    def get_fast_apis(self, rabbit_host):
        return [NerApi(self, rabbit_host)]