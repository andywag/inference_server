
from bert_interface_wrapper import BertInterfaceWrapper

import numpy as np
import attr
from dataclasses import dataclass
from model_proto import ModelProto, SignalProto
from typing import List
from bert_proto import *

logits = SignalProto('logits',np.zeros(shape=3456, dtype=np.float32))
from general_fastapi import BasicFastApi
from api_classes import Ner, NerResult, NerResponse
from transformers import BertTokenizerFast
import time

class NerApi(BasicFastApi):        
    def __init__(self, proto):
        super().__init__(proto, 'ner')
        self.input_type = Ner
        self.output_type = NerResponse
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
    
    def create_input(self, ner:Ner, triton_input):
        result = self.tokenizer.encode_plus(text = ner.text, max_length=384, 
            padding='max_length',return_offsets_mapping=True)
        full_sum = np.sum(np.asarray(result['attention_mask'],dtype=np.uint32))
        input_ids = np.asarray([result['input_ids']],dtype=np.uint32)
        token_type_ids = np.asarray([[full_sum]],dtype=np.uint32)
        offset_mapping = result['offset_mapping']

        triton_input[0].set_data_from_numpy(input_ids)
        triton_input[1].set_data_from_numpy(token_type_ids)
        triton_input[2].set_data_from_numpy(np.asarray([[0]],dtype=np.uint64))
        
        return full_sum, offset_mapping

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
    max_batch_size:int = 512
    checkpoint:str ='https://huggingface.co/dslim/bert-large-NER/resolve/main/pytorch_model.bin'
    inputs:List[SignalProto] = [input_ids, segment_ids, query_ids]
    outputs:List[SignalProto] = [query_ids_result, logits]

    def create_model(self):
        base_path = "./bert"
        config = f"{base_path}/configs/sut_inference_pack_384_ner_single.json"
        model = BertInterfaceWrapper( 
            config=config,
            ner=True)
        return model

    def get_fast_apis(self):
        return [NerApi(self)]