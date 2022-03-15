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
from   packing_utils import pack_data
import copy
from heap_utils_triton import CircularInputBuffer, PackingQueueHolderNew, CircularOutputBuffer

import traceback


import logging
log = logging.getLogger()
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)



class BertInterfaceWrapper():
    def __init__(self, 
            args, 
            config, 
            input_data_queue,
            internal_buffer_width = 256, 
            input_queue_size = 256, 
            input_queue_start = 6):
        
        self.input_data_queue = input_data_queue
  
        # Convenience Varaible to handle callback ordering
        self.first_tensor_id = None


        try:
            self.runner,  self.internal_args = bi.create_model_external(config, {})
        except Exception as e:
            log.error("Failed to Create", e)


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
        self.blocks = PackingQueueHolderNew(self.b, self.s, internal_buffer_width=internal_buffer_width, input_queue_size=input_queue_size, 
            input_queue_start=input_queue_start)

        # Object used to report statistics
        self.stepio = popart.PyStepIOCallback(self.get_input,
                        self.get_input_complete,
                        self.get_output,
                        self.get_output_complete)

        self.run_enable = True
        # Thread to run the IPU in parallel to the input and output process
        self.run_process = threading.Thread(target=self.run_ipu, args=())
        self.run_process.start()

        self.output_count = 0
        # Thread to run the outptu process in parallel to the input and run
        self.out_process = threading.Thread(target = self.output_results, args = (self.final_output_queue, self.b, self.s))
        self.out_process.start()   
   
        self.rx_data_process = threading.Thread(target=self.async_handle_input, args= ())
        self.rx_data_process.start()

    # Run task for IPU using the StepIO callback
    # This function runs for the full batches per step
    # Individual batches are handled in the runner callbacks functions
    def run_ipu(self):
        step = self.stepio
        while self.run_enable:
            tic = time.time()
            self.runner.session.run(step)
            runtime = time.time()
            print("Running Time", runtime - tic)
       
        print("Done Running IPU")


    def output_results(self, input_queue, block_size, hidden):
        while True:
            results_total = input_queue.get()
            if self.run_enable:
                done = self.output_results_internal(results_total, block_size, hidden)
                if done:
                    self.run_enable = False
                 
    def output_results_internal(self, results_total, block_size, hidden):
        tic = time.time()
        results = results_total[0] 
        specs = results_total[1]
    
        results = np.reshape(results,(block_size,hidden,2)).astype(np.float32)
        logits = results.reshape((block_size,2*hidden))

        #log.info(f"Results {specs[0].sender} {len(specs)}")
        for spec in specs:
            real_logits = -10.0*np.ones(2*hidden).astype(np.float32)
            real_logits[:2*spec.l] = logits[spec.row,2*spec.col:2*spec.col + 2*spec.l].reshape(2*spec.l)
            spec.sender.put((spec.id,real_logits))
      

    def async_handle_input(self):
        while True:
            transfer = self.input_data_queue.get()
            #self.blocks.insert( transfer[0], transfer[1][0],transfer[2], transfer[3], False)
            self.blocks.insert_block(transfer)

    def get_input(self, tensor_id, prefetch):
        if self.first_tensor_id is None:
            self.first_tensor_id = tensor_id
        if tensor_id == self.first_tensor_id:
            block = self.blocks.get_legend()
           
            if block is not None:
                self.run_output_queue.put(block)
            else:
                self.finished_input = True

        data = self.blocks.get_tensor(tensor_id)
        return data
       
    def get_input_complete(self, tensor_id):
        pass
           
    def get_output_complete(self, tensor_id):
        self.output_buffer.get_complete()


    def get_output(self, tensor_id: str):
        data = self.output_buffer.get(tensor_id)
        return data
    
    def flush_queries(self):
        print("flush queries")

    def process_latencies(self, latencies_ns):
        print("Handle Latency")

    def __del__(self):
        print("Finished destroying SUT.")


