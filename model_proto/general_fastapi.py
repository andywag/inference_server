
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dataclasses import dataclass
from model_proto import ModelProto
from project_proto import ProjectProto

from general_client import GeneralClient
from typing import TypeVar
import time

import tritonclient.grpc as grpcclient
import dataclasses

class BasicFastApi:
    
    def __init__(self, proto:ModelProto, path:str):
        self.path = path
        self.proto = proto
        
        self.client = GeneralClient(self.proto)
        self.grpc_client = grpcclient.InferenceServerClient("localhost:8001")

        self.inputs, self.outputs = self.client.inputs, self.client.outputs


    def run(self, model_input):
        tic = time.time()
        state = self.create_input(model_input, self.inputs)
        response = self.grpc_client.infer(self.proto.name,
                        self.inputs,
                        request_id=str(0),
                        outputs=self.outputs)
        internal_result = self.handle_output(response, state, tic)
        return dataclasses.asdict(internal_result)
         

    def get_api(self):
        return self



    def create_input(self, model_input, client_input):
        raise NotImplementedError("")

    def handle_output(self, response, state, tic):
        raise NotImplementedError("")


def get_apis(prototype):
    apis = []
    for model in prototype.models:
        #print("A", model.name)
        for api in model.get_fast_apis():
            print("Creating", api)
            apis.append(api.get_api())
    return apis

@dataclass
class GeneralFastApi:
    prototype:ProjectProto
    app:FastAPI = FastAPI()

    def create_apis(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=['*'],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        for model in self.prototype.models:
            for api in model.get_fast_apis():
                api.create_api(self.app)