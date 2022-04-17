
from dataclasses import dataclass
from model_proto import ModelProto
from typing import Optional, List
import time
import threading

@dataclass
class GeneralClient:
    proto:ModelProto
    batch_size:Optional[int] = 1

    #def __post_init__(self):
    #    self.client = grpcclient.InferenceServerClient("localhost:8001")
    #    self.create_basic_request()
    #    self.create_basic_call()
        

    def create_basic_call(self):
        for i, signal in enumerate(self.proto.inputs):
            np_data = signal.get_random_signal(self.batch_size)
            self.inputs[i].set_data_from_numpy(np_data)

    def create_basic_request(self):
        self.inputs, self.outputs = [], []
        for signal in self.proto.inputs:
            
            self.inputs.append(grpcclient.InferInput(signal.name, signal.get_shape(self.batch_size), 
                np_to_triton_dtype(signal.signal.dtype)))

        for signal in self.proto.outputs:
            self.outputs.append(grpcclient.InferRequestedOutput(signal.name))
        
    
    def single_packet(self, batch_size:int=16, sim_length:int = 65536):
        real_sim_length = int(sim_length/batch_size)
        update_length = int(4096/batch_size)
        tic = time.time()
        for x in range(real_sim_length):
            self.client.infer(self.proto.name,
                                self.inputs,
                                request_id=str(1),
                                outputs=self.outputs)
            if x % update_length == update_length - 1:
                print("Response", x, time.time() - tic, update_length*batch_size/(time.time() - tic))
                tic = time.time()

    def parallel_packet(self, batch_size:int=16, sim_length:int = 65536, threads:int = 4):
        print("Running Parallel Test", self.proto.name, batch_size, threads, sim_length)
        for _ in range(threads):
            thread = threading.Thread(target=self.single_packet, args=(batch_size, sim_length))
            thread.start()

