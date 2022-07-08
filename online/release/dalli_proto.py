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
        super().__init__(proto, host, 'dalli')
        self.input_type = Dalli
        self.output_type = DalliResponse
    
    def create_rabbit_input(self, dalli:Dalli):
        return DalliInput(dalli.text, dalli.seed), None

    def handle_rabbit_output(self, response, state, tic):
        image = response['result']
        #print("KKKK", image)
        #image = [[[1]]]
        return DalliResponse(image, time.time() - tic)

@dataclass 
class DalliInput:
    text:str
    seed:int

    def items(self):
        return [self.text, self.seed]

@dataclass
class DalliOutput:
    result:List[List[int]]

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