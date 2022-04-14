
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import os
import multiprocessing as mp
import time
import threading
import shutil

from triton_interface import ClientSideInterface, ServerSideInterface
from model_proto import ModelProto
from general_client import GeneralClient

BASE_PORT = 50000



def create_dir(location):
    if not os.path.exists(location):
        os.makedirs(location)


@dataclass
class ProjectProto:
    """ Project Description : Class to hold a list of running models """
    name:str
    models:List[ModelProto]
    models_dict:Dict[str,ModelProto]

    def create_triton_structure(self, path:str):
        """ Creates the triton structure for the project"""
        template_path = os.path.dirname(os.path.realpath(__file__)) + "/triton_model_template.py"
        create_dir(path)
        [x.create_triton_structure(path) for x in self.models]

    def run_ipus(self):
        """ Run all of the models for this project in a separate process"""
        self.processes = [mp.Process(target = x.run_ipu, args=(BASE_PORT+16*i,)) for i, x in enumerate(self.models)]
        [x.start() for x in self.processes]

        # Keep things alive
        while True:
            time.sleep(10)

    def _get_model(self, model_name):
        for m in self.models:
            if m.name == model_name:
                return m
        return None

    def get_triton_interface(self, full_model_name):
        """ Return the triton interface based on name. Called from triton wrapper. """
        def get_model_index(model_name, num_back=4):            
            model_indices = model_name.split("_")
            model_name = model_indices[0]
            model_base, model_index = 0, 0
            if len(model_indices) == 3:
                model_base = int(model_indices[2]) % num_back
                model_index = int(int(model_indices[2])/num_back)
            return model_name, model_base, model_index

        model_name, model_base, model_index = get_model_index(full_model_name)

        for i, model in enumerate(self.models):
            if model.name == model_name:
                return model.get_triton_interface(BASE_PORT+16*i+model_base, model_base, model_index)

    def single_client_test(self, model_name, threads=4,batch_size=16, sim_length=65536):
        client = GeneralClient(self._get_model(model_name), batch_size=batch_size)
        client.parallel_packet(batch_size=batch_size, sim_length=sim_length,threads=threads)

