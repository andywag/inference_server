
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from dataclasses import dataclass
from model_proto import ModelProto
from project_proto import ProjectProto

from typing import TypeVar
import time

import dataclasses
import queue

from rabbit_run_queue import RabbitRunQueue
from dacite import from_dict
import json

class BasicFastApi:
    
    def __init__(self, proto:ModelProto, path:str):
        self.path = path
        self.proto = proto
        

        self.rabbit_queue = RabbitRunQueue(proto.name)

    def run_rabbit(self, model_input):
        tic = time.time()
        # Create a queue to hold the output
        data_queue = queue.Queue()
        def callback(result):
            data_queue.put(result)
        
        # Create the rabbit input and convert it to json to put into rabbit message queue
        model_input, state = self.create_rabbit_input(model_input)
        model_input_dict = dataclasses.asdict(model_input)
        model_input_json = json.dumps(model_input_dict)

        # Send the message to a rabbit message queue
        # FIXME : time Hack for UUID
        self.rabbit_queue.post_message(model_input_json, str(time.time()), callback)

        # Handle the output of the model which is attached to the message queue
        result = data_queue.get()
        result_dict = json.loads(result)
        result = self.handle_rabbit_output(result_dict, state, tic)
        print("Result", result)
        return result


    def get_api(self):
        return self

    def create_rabbit_input(self, ner:Ner):
        """ Method which converts the input from FAST API to the model input """
        raise NotImplementedError("")

    def handle_rabbit_output(self, response, state, tic:int):
        """ Method to convert the output of the model back to the general interface """
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