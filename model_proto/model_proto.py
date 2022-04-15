
from dataclasses import dataclass
from typing import List, Optional, TypeVar
import numpy as np
import os
import multiprocessing as mp
import time
import threading
import shutil
from dacite import from_dict


from rabbit_client import RabbitProtoWrapper
import traceback

def create_dir(location):
    if not os.path.exists(location):
        os.makedirs(location)
        
@dataclass
class ModelProto:
    """ Model Description : Description of an inference model
        Args:
            name (str) : Name of the model
            checkpoint (str) : Path to the model checkpoint
            input_type : Type of Input (dataclass)
            output_type : Type of Output (dataclass)
    """
    name:str
    checkpoint:str

    input_type:TypeVar=None 
    output_type:TypeVar=None

    def body_to_input(self, body:dict):
        print("Base", self.input_type)
        user = from_dict(data_class=self.input_type, data=body)
        return user

    def create_model():
        """ Method used to create and run a model """
        raise NotImplementedError("ModelProto needs create_model defined")
 

    def _run_rabbit(self, config):
        """ Run the model side interface """
        try:
            interface = RabbitProtoWrapper(self, config)
            interface.run()
        except Exception as e:
            print("Rabbit Failed : ", e)
            traceback.print_exc()

    def run_ipu(self, config):
        self.model = self.create_model()

        rabbit_thread = threading.Thread(target=self._run_rabbit, args=(config,))
        rabbit_thread.start()

      
        while True:
            time.sleep(10)
        #print("Finished")


    def get_fast_apis(self):
        return []
        


    