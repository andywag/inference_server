
import time
import sys
import numpy as np
import queue
import zmq
import threading
import traceback
import multiprocessing as mp

import triton_python_backend_utils as pb_utils

import logging
log = logging.getLogger(__name__)
#logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.ERROR)

class ModelWrapper:
    def __init__(self, model_name = "bert", model_base = None, 
        model_index = None, num_back = None, base_port = 50016):

        print("Creating Model Wrapper", model_name)
        self.model_name = model_name
        # Number of Possible Threads
        if num_back is None:
            self.num_back = 4
        else:
            self.num_back = num_back
        # Separate Output Queues
        self.model_base = model_base
        self.model_index = model_index

        self.base_port = base_port + self.model_index*self.num_back + self.model_base
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect ("tcp://localhost:%s" % self.base_port)

         

    def transfer(self, request):

        self.socket.send(pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy(),zmq.SNDMORE)
        self.socket.send(pb_utils.get_input_tensor_by_name(request, "segment_ids").as_numpy(),zmq.SNDMORE)
        self.socket.send(pb_utils.get_input_tensor_by_name(request, "query_ids").as_numpy())
                #print("Sending Packet", port)
        id_group = self.socket.recv()
        logits = self.socket.recv()

        id_group = np.frombuffer(id_group, dtype=np.uint64)
        logits = np.frombuffer(logits, dtype=np.float32)
        logits = logits.reshape(int(len(logits)/3456),3456)
        return id_group, logits

       
    def execute(self, requests):

        id_group, logits_response = self.transfer(requests[0])

        logits = pb_utils.Tensor("logits", logits_response)
        query_return_np = np.asarray(id_group, dtype=np.uint64)
        query_ids_result = pb_utils.Tensor("query_ids_result", query_return_np)
        inference_response = pb_utils.InferenceResponse(output_tensors=[query_ids_result, logits])
        #print("Response", time.time() - tic)

        return [inference_response]
