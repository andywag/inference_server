
from dataclasses import dataclass
from typing import List, Optional
import numpy as np
import os
import multiprocessing as mp
import time
import threading
import shutil

from signal_proto import SignalProto

from triton_interface import ClientSideInterface, ServerSideInterface

def create_dir(location):
    if not os.path.exists(location):
        os.makedirs(location)
        
@dataclass
class ModelProto:
    """ Model Description : Description of an inference model
        Args:
            name (str) : Name of the model
            max_batch_size (int) : Maximum batch size supported
            checkpoint (str) : Path to the model checkpoint
            inputs : List of Inputs to the model
            outputs : List of Outputs from the model
            backends : Number of Parallel backends running to emulate asynchronous operation
    """
    name:str
    max_batch_size:int
    checkpoint:str
    inputs:List[SignalProto]
    outputs:List[SignalProto]
    backends:int = 4

    def _create_triton_config(self, path:str):
        """ Create the python config.pbtxt file """
        config_path = f"{path}/config.pbtxt"
        with open(config_path,'w') as fptr:
            fptr.write(f'name: "{self.name}"\n')
            fptr.write(f'backend: "python"\n')
            fptr.write(f'max_batch_size: {self.max_batch_size}\n')
            fptr.write('\n')
            [x.create_signal(fptr, 'input') for x in self.inputs]
            [x.create_signal(fptr, 'output') for x in self.outputs]
            fptr.write("\n")
            fptr.write("instance_group [{\n") 
            fptr.write("   count: 4\n")
            fptr.write("   kind: KIND_CPU\n") 
            fptr.write("}]\n")

    def create_model():
        """ Method used to create and run a model """
        raise NotImplementedError("ModelProto needs create_model defined")
 


    def create_triton_structure(self, path):
        """ Create the triton structure """
        model_dir = f"{path}/{self.name}"
        python_dir = f"{model_dir}/1"
        create_dir(model_dir)
        create_dir(python_dir)
        template_path = os.path.dirname(os.path.realpath(__file__)) + "/triton_model_template.py"
        shutil.copyfile(template_path, f"{python_dir}/model.py")

        self._create_triton_config(model_dir)

        
    def _run_client(self, run_port:int):
        """ Run the model side interface """
        try:
            interface = ClientSideInterface(self, run_port)
            interface.run()
        except Exception as e:
            print("Client Failed", e)


    def run_ipu(self, port):
        
        print(f"Running Model {self.name} on {port}")
        self.model = self.create_model()

        for x in range(self.backends):
            worker_thread = threading.Thread(target = self._run_client, args=(port+x,))
            worker_thread.start()
        while True:
            time.sleep(10)
        print("Finished")

    def get_triton_interface(self, port, base, index):
        """ Run the Triton Side Interface """
        triton_interface = ServerSideInterface(self, port, base, index)
        return triton_interface


    def get_fast_apis(self):
        return []
        


    