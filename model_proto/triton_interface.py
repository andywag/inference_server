

import zmq
from dataclasses import dataclass
#from model_proto import ModelProto
import numpy as np
import queue
import time

import triton_python_backend_utils as pb_utils

import logging
log = logging.getLogger(__name__)

class ServerSideInterface:

    def __init__(self, model_proto, port, model_base, model_index):

        self.model_proto = model_proto
        
        self.base_port = port #50000 + model_base
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect ("tcp://localhost:%s" % self.base_port)
        print(f"Running {model_proto.name} {model_base} {model_index} : {self.base_port}")

        
    def execute(self, requests):
        #print("Running Server Side")
        for x in range(len(self.model_proto.inputs)):
            signal = self.model_proto.inputs[x]
            data = pb_utils.get_input_tensor_by_name(requests[0], signal.name).as_numpy()
            batch_size = data.shape[0]
            if x == len(self.model_proto.inputs)-1:
                self.socket.send(data)
            else:
                
                self.socket.send(data,zmq.SNDMORE)

        output_tensors = []
        for x in range(len(self.model_proto.outputs)):
            signal = self.model_proto.outputs[x]
            result = self.socket.recv()
            np_result = np.frombuffer(result, dtype=signal.signal.dtype)
            new_dim = list(signal.signal.shape)
            new_dim.insert(0,batch_size)
            np_result = np_result.reshape(new_dim)
            # FIXME : Might require reshape to original size
            tensor_result = pb_utils.Tensor(signal.name, np_result)
            output_tensors.append(tensor_result)
        return [pb_utils.InferenceResponse(output_tensors=output_tensors)]




class ClientSideInterface:
    def __init__(self, model_proto, port:int):
        self.model_proto = model_proto
        self.result_queue = queue.Queue()

        #print("Creating Model Side", port)
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        #self.socket.connect ("tcp://localhost:%s" % self.base_port)
        self.socket.bind("tcp://*:%s" % port)
        #print(f"Opened Client {model_proto.name} : {port}")
        #time.sleep(20)

    def run(self):

        def callback(result):
            self.result_queue.put(result)
        
        while True:
            inputs = []
            for x in range(len(self.model_proto.inputs)):
                receive_data = self.socket.recv()
            #    print("Received Data")
                signal = self.model_proto.inputs[x]
                np_receive_data = np.frombuffer(receive_data, dtype=signal.signal.dtype)
                batch_size = int(len(np_receive_data)/signal.signal.size)
                # FIXME : Properly Support multiple dimensions
                np_receive_data = np_receive_data.reshape((batch_size,signal.signal.size))
                inputs.append(np_receive_data)

            self.model_proto.model.run_data(inputs, callback)
            results = []
            for x in range(batch_size):
                results.append(self.result_queue.get())


            for x in range(len(self.model_proto.outputs)):
                signal = self.model_proto.outputs[x]
                new_dim = list(signal.signal.shape)
                new_dim.insert(0,batch_size)
                np_result = np.zeros(shape=tuple(new_dim),dtype=signal.signal.dtype)

                for y in range(len(results)):
                    np_result[y,:] = results[y][x]

                if x == len(self.model_proto.outputs)-1:
                    self.socket.send(np_result)
                else:
                    self.socket.send(np_result, zmq.SNDMORE)


            


