import sys
sys.path.append("./min-dalle_int")

import numpy as np
import attr
from dataclasses import dataclass
from typing import List
from model_proto import *



from general_fastapi import BasicFastApi
from api_classes import Dalli, DalliResponse
#import torch

import time


class DalleApi(BasicFastApi):        
    def __init__(self, proto, host):
        super().__init__(proto, host, 'dalle')
        self.input_type = Dalli
        self.output_type = DalliResponse
    
    def create_rabbit_input(self, dalli:Dalli):
        return DalliInput(dalli.text, dalli.seed), None

    def handle_rabbit_output(self, response, state, tic):
        image = response['result']
        return DalliResponse(image, time.time() - tic)

@dataclass 
class DalliInput:
    input_ids:List[List[int]]
    input_length:List[List[int]]
    output_length:List[List[int]]

    def items(self):
        return [self.input_ids, self.input_length[0], self.output_length[0]]

@dataclass
class DalliOutput:
    result:List[List[float]]

    @staticmethod
    def create(data:np.ndarray):
        return DalliOutput(data.tolist())


@dataclass
@attr.s(auto_attribs=True)
class DalliProto(ModelProto):
    name:str = "dalli"
    checkpoint:str ='None'

    input_type = DalliInput
    output_type = DalliOutput

    def create_model(self):
        from min_dalli_interface import MinDalleInterfaceWrapper
        model = MinDalleInterfaceWrapper() 
        return model

    def get_fast_apis(self, host):
        return [DalleApi(self, host)]