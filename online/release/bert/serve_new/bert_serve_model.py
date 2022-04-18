# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import time
import sys

import sys

from bert_interface_wrapper import BertInterfaceWrapper

import numpy as np
import queue
import zmq
import traceback


import logging
log = logging.getLogger()
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)




class BertServeModel:
    def __init__(self, ner=False ):
        self.ner = ner
        self._create_bert()
        self.result_queue = queue.Queue()


    def handle_zmq_receive(self, socket):
        input_ids = socket.recv()
        segment_ids = socket.recv()
        query_ids = socket.recv()
        input_ids = np.frombuffer(input_ids, dtype=np.int32)

        input_ids = input_ids.reshape( (int(len(input_ids)/384), 384))
        segment_ids = np.frombuffer(segment_ids, dtype=np.int32).reshape(len(input_ids), 1)
        query_ids = np.frombuffer(query_ids, dtype=np.uint64)
        return (input_ids, segment_ids, query_ids)

    def handle_zmq_transmit(self, socket, output_data):
        socket.send(np.asarray(output_data[0], dtype=np.uint64), zmq.SNDMORE)
        socket.send(output_data[1])


    def run(self, input_data):
        def callback(result):
            self.result_queue.put(result)

        #print("Handle Input")
        transmit_length = len(input_data[0])   
        self.bert.run_data(input_data, callback)
        #print("Finish INput")
        
        result =  self.handle_output(transmit_length)
        #print("Done")
        return result

    def handle_output(self, transmit_length):
        if self.ner:
            logits_response = -10*np.ones((transmit_length,9*self.bert.internal_args.sequence_length),dtype=np.float32)
        else:
            logits_response = -10*np.ones((transmit_length,2*self.bert.internal_args.sequence_length),dtype=np.float32)

        id_group = []

        for x in range(transmit_length):
           
            ids, result = self.result_queue.get()
            id_group.append(ids)
            logits_response[x,:] = result
        
        return (id_group, logits_response)


    def _create_bert(self):
        base_path = "/localdata/andyw/projects/public_examples/applications/inference/bert"
        

        config = f"{base_path}/configs/sut_inference_pack_384_torch_single.json"
        if self.ner:
            config = f"{base_path}/configs/sut_inference_pack_384_ner_single.json"

        print("Calling Bert Creation")
        self.bert = BertInterfaceWrapper( 
            config=config,
            ner=self.ner)
      






    



