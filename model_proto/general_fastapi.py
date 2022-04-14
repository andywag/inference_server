
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
import queue

from rabbit_run_queue import RabbitRunQueue
from dacite import from_dict
import json

class BasicFastApi:
    
    def __init__(self, proto:ModelProto, path:str):
        self.path = path
        self.proto = proto
        
        self.client = GeneralClient(self.proto)
        self.grpc_client = grpcclient.InferenceServerClient("localhost:8001")

        self.inputs, self.outputs = self.client.inputs, self.client.outputs

        self.rabbit_queue = RabbitRunQueue(proto.name)

    def run_rabbit(self, model_input):
        tic = time.time()
        data_queue = queue.Queue()
        def callback(result):
            data_queue.put(result)
        model_input, state = self.create_rabbit_input(model_input)
        model_input_dict = dataclasses.asdict(model_input)
        model_input_json = json.dumps(model_input_dict)
        # FIXME : time Hack for UUID
        self.rabbit_queue.post_message(model_input_json, str(time.time()), callback)
        result = data_queue.get()
        result_dict = json.loads(result)
        result = self.handle_rabbit_output(result_dict, state, tic)
        print("Result", result)
        return result


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

def create_rabbit_queues(prototype):
    queue_map = dict()
    for model in prototype.models:
        rabbit_queue = RabbitRunQueue(model.name)
        queue_map[model.name] = rabbit_queue
    return queue_map

def get_apis(prototype):
    apis = []
    for model in prototype.models:
        for api in model.get_fast_apis():
            print("Creating", api)
            apis.append(api.get_api())
    return apis

def get_apis_dict(prototype):
    apis = dict()
    for key in prototype.models_dict.keys():
        apis[key] = prototype.models_dict[key].get_fast_apis()[0]
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