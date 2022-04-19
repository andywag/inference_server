
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import os
import multiprocessing as mp
import time
import threading
import shutil

from model_proto import ModelProto

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

    def run_ipus(self, config):
        """ Run all of the models for this project in a separate process"""
        self.processes = [mp.Process(target = x.run_ipu, args=(config,)) for i, x in enumerate(self.models)]
        [x.start() for x in self.processes]

        # Keep things alive
        while True:
            time.sleep(10)

    def _get_model(self, model_name):
        for m in self.models:
            if m.name == model_name:
                return m
        return None

    

