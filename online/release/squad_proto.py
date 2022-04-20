
from bert_proto import *

import numpy as np
import attr
from dataclasses import dataclass
from typing import List, TypeVar
from bert_proto import *
from transformers import BertTokenizerFast
import time
import queue
from split_squad import get_predictions
from general_fastapi import BasicFastApi


from api_classes import Squad, SquadArray, SquadResult, SquadResponse

@dataclass
class SquadOutput:
    logits:List[float]

class SquadApi(BasicFastApi): 
    input_type = Squad
    output_type = SquadResponse 
    path = '/squad'      
    def __init__(self, proto, host):
        super().__init__(proto, host, 'squad')
        self.input_type = Squad
        self.output_type = SquadResponse
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-large-uncased")
    
    def create_rabbit_input(self, squad:Squad):
        result = self.tokenizer.encode_plus(text = squad.question, text_pair = squad.answer, 
            max_length=384, padding='max_length',return_offsets_mapping=True)
        
        slen = np.sum(np.asarray(result['attention_mask'],dtype=np.uint32))
        extra = np.sum(np.asarray(result['token_type_ids'],dtype=np.uint32))

        input_ids = np.asarray([result['input_ids']],dtype=np.uint32)
        token_type_ids = np.asarray([[slen-extra]],dtype=np.uint32)
        offset_mapping = result['offset_mapping']

        print("Input Ids", input_ids.shape)

        return BertInput(input_ids.tolist(), token_type_ids.tolist(), [0]), (offset_mapping)

    def handle_rabbit_output(self, response, state, tic:int):
        
        offset_mapping = state
        logits = np.asarray(response['logits'])
        real_logits = logits.reshape([384,2])

        results = get_predictions(real_logits[:,0], real_logits[:,1])
        n_results = []

        for r in results:                
            n_result = SquadResult(text="", 
                logits_sum=float(r.logit), 
                start=offset_mapping[r.start_index][0], 
                end=offset_mapping[r.end_index][1])
                
            n_results.append(n_result)
        print("Handle Response", n_results)

        squad_response = SquadResponse(n_results, time.time() - tic)
        return squad_response


@dataclass
@attr.s(auto_attribs=True)
class SquadProto(ModelProto):
    name:str = "squad"
    checkpoint:str ='https://huggingface.co/bert-large-uncased-whole-word-masking-finetuned-squad/resolve/main/pytorch_model.bin'

    input_type = BertInput
    output_type = NerOutput

    def create_model(self):
        from bert_interface_wrapper import BertInterfaceWrapper

        print("Creating Model")
        base_path = "./bert"
        config = f"{base_path}/configs/sut_inference_pack_384_torch_single.json"
        model = BertInterfaceWrapper( 
            config=config,
            ner=False)
        return model

    def get_fast_apis(self, host):
        return [SquadApi(self, host)]



