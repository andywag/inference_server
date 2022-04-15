# Copyright (c) 2019 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import array
import json
import os
import sys


import numpy as np
import time
import bert_inference as bi
import popart
import math
import threading 
import queue
import copy
from heap_utils_simple import CircularInputBuffer, PackingQueueHolder, CircularOutputBuffer

import traceback


import logging
log = logging.getLogger()
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)

class Temp:
    pass


class BertInterfaceWrapper():
    def __init__(self, 
            config, 
            internal_buffer_width = 512, 
            input_queue_size = 512, 
            input_queue_start = 6,
            ner=False):
        
        self.ner = ner
        self.input_data_queue = queue.Queue(maxsize=2048)

  
        # Convenience Varaible to handle callback ordering
        self.first_tensor_id = None


        try:
            self.runner,  self.internal_args = bi.create_model_external(config, {})
        except Exception as e:
            log.error("Failed to Create", e)
            sys.exit(0)


        # Size Parameters for the System
        # w : Width of Internal Loading Buffer
        # b : Total Batch Size 
        # s : Maximum Sequence Length

        w = internal_buffer_width
        self.s = self.internal_args.sequence_length
        self.b = self.internal_args.micro_batch_size
    
        rep = self.internal_args.replication_factor

        # Loading Queue to Store the data : Block Information is written inside the queue until use
        # Used to setup blocking
        # Queue which stores the block information for the current run
        self.run_output_queue = queue.Queue()
        # Queue which contains the output information and result
        self.final_output_queue = queue.Queue()    

        # Buffer to Store the Data Used by the Design
        
        # Buffer to Store the Output Data 
        self.output_buffer = CircularOutputBuffer(self.internal_args.batches_per_step, self.runner.anchors, self.run_output_queue, 
            self.final_output_queue, self.internal_args.replication_factor)
        # Instance used to pack the input data
        self.packing_queue = PackingQueueHolder(self.b, self.s, internal_buffer_width=internal_buffer_width, input_queue_size=input_queue_size, 
            input_queue_start=input_queue_start)

        # Object used to report statistics
        self.stepio = popart.PyStepIOCallback(self.get_input,
                        self.get_input_complete,
                        self.get_output,
                        self.get_output_complete)

        self.run_enable = True
        # Thread to run the IPU in parallel to the input and output process
        self.run_process = threading.Thread(target=self._run_ipu_thread, args=())
        self.run_process.start()

        # Thread to run the outptu process in parallel to the input and run
        self.out_process = threading.Thread(target = self._output_result_thread, args = (self.final_output_queue, self.b, self.s))
        self.out_process.start()   
   
        self.rx_data_process = threading.Thread(target=self._input_thread, args= ())
        self.rx_data_process.start()



    # Thread which runs the IPU
    def _run_ipu_thread(self):
        step = self.stepio
        while True:
            tic = time.time()
            self.runner.session.run(step)
            runtime = time.time()
            print("Running Time", runtime - tic, self.packing_queue.run_block_queue.qsize())
       
        print("Done Running IPU")

    # Thread which outputs the results
    def _output_result_thread(self, input_queue, block_size, hidden):
        while True:
            results_total = input_queue.get()
            if self.run_enable:
                if self.ner:
                    self.output_results_internal_ner(results_total, block_size, hidden)
                else:
                    self.output_results_internal(results_total, block_size, hidden)

    # Thread to handle the input data
    def _input_thread(self):
        while True:
            transfer = self.input_data_queue.get()
            self.packing_queue.insert_block(transfer)        

     
    def run_data(self, input_data, callback):
        self.input_data_queue.put((input_data, callback))


    
    def output_results_internal(self, results_total, block_size, hidden):
        results = results_total[0] 
        specs = results_total[1]
    
        results = np.reshape(results,(block_size,hidden,2)).astype(np.float32)
        logits = results.reshape((block_size,2*hidden))

        for spec in specs:
            real_logits = -10.0*np.ones(2*hidden).astype(np.float32)
            real_logits[:2*spec.l] = logits[spec.row,2*spec.col:2*spec.col + 2*spec.l].reshape(2*spec.l)
            spec.sender((spec.id,real_logits))
      
    def output_results_internal_ner(self, results_total, block_size, hidden):
        results = results_total[0] 
        specs = results_total[1]
    
        results = np.reshape(results,(block_size,hidden,9)).astype(np.float32)
        logits = results.reshape((block_size,9*hidden))
        
        for spec in specs:
            real_logits = -10.0*np.ones(9*hidden).astype(np.float32)
            real_logits[:9*spec.l] = logits[spec.row,9*spec.col:9*spec.col + 9*spec.l].reshape(9*spec.l)
            spec.sender((spec.id,real_logits))




    def get_input(self, tensor_id, prefetch):
        if self.first_tensor_id is None:
            self.first_tensor_id = tensor_id
        if tensor_id == self.first_tensor_id:
            block = self.packing_queue.get_legend()
           
            if block is not None:
                self.run_output_queue.put(block)
            else:
                self.finished_input = True

        data = self.packing_queue.get_tensor(tensor_id)
        return data
       
    def get_input_complete(self, tensor_id):
        pass
           
    def get_output_complete(self, tensor_id):
        self.output_buffer.get_complete()


    def get_output(self, tensor_id: str):
        data = self.output_buffer.get(tensor_id)
        return data
    

