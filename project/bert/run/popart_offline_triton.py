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

import mlperf_loadgen as lg
import numpy as np
import torch
from   squad_QSL import get_squad_QSL
import time
import bert_inference as bi
import popart
import math
import threading 
import queue
from  packing_utils_triton import pack_data_triton_queue, empty_data_transfer
import time
import asyncio
import math
from collections import deque



class BertInterfaceWrapper:
    def __init__(self,  
            args, 
            config, 
            input_data_queue,
            output_data_queues):
       
        print("Creating Bert Model")
        self.runner,  self.internal_args = bi.create_model_external(config, dict())

        self.input_data_queue = input_data_queue
        self.output_data_queues = output_data_queues

        self.done = True

        # Queue which contains input data to be run
        self.run_queue = queue.Queue(maxsize=1)
        # Queue which contains output data from the IPU Run
        self.out_queue = queue.Queue(maxsize=100)    
        # Queue which contains output data from the IPU Run
        #self.final_queue = queue.Queue()
        self.length_queue = queue.Queue()


        # Running thread to split the IPU run from the input and the output
        self.run_started = False
        self.run_process = threading.Thread(target=self.run_ipu, args= (self.run_queue, self.out_queue))
        self.run_process.start()

        # Output thread to handle the results of the run
        hidden = self.internal_args.sequence_length
        block_size = self.internal_args.micro_batch_size*self.internal_args.batches_per_step*self.internal_args.replication_factor
        self.b = block_size
        self.s = hidden

        # Thread to run the outptu process in parallel to the input and run
        self.out_process = threading.Thread(target = self.output_results, args = (self.out_queue, self.b, self.s))
        self.out_process.start()       


        self.rx_data_process = threading.Thread(target=self.async_handle_input, args= ())
        self.rx_data_process.start()

        # Dictionary containing an index of the query associated with it's callback
        self.response_dict = dict()
        self.output_count = 0

        self.run_enable = True
   
    def async_handle_input(self):
        sequence_length = self.internal_args.sequence_length
        block_size = self.internal_args.micro_batch_size*self.internal_args.batches_per_step
        count = 0
        last_value = None
        while True:
            transfer, last_value = pack_data_triton_queue(self.input_data_queue, block_size, sequence_length, last = last_value)
            count += len(transfer.specs)
            print("Transfer", count, self.input_data_queue.qsize(), self.run_queue.qsize())
            self.run_queue.put(transfer)
            #self.length_queue.put(transfer.count)

    def run_ipu(self, input_queue, output_queue):
        while True:
            if self.run_started and  input_queue.qsize() == 0:
                input_data = empty_data_transfer(self.b, self.s)
                self.run_started = False
            else:
                input_data = input_queue.get()
                self.run_started = True
            
            tic = time.time()
            results = self.runner.run(input_data.data)
            result =  input_data.update(results)
            runtime = time.time()
            print("Running Time", runtime - tic)
            output_queue.put(result)
        

    def output_results(self, input_queue, block_size, hidden):
        while True:
            results_total = input_queue.get()
            if True: #self.run_enable:
                done = self.output_results_internal(results_total, block_size, hidden)
                if done:
                    self.run_enable = False
                 
    def output_results_internal(self, results_total, block_size, hidden):
        tic = time.time()
        results = results_total[0] 
        specs = results_total[1]
    
        results = np.reshape(results,(block_size,hidden,2)).astype(np.float32)
        logits = results.reshape((block_size,2*hidden))

        for spec in specs:
            real_logits = -10.0*np.ones(2*hidden).astype(np.float32)
            real_logits[:2*spec.l] = logits[spec.row,2*spec.col:2*spec.col + 2*spec.l].reshape(2*spec.l)
            self.output_data_queues[spec.sender].put((spec.id,real_logits))



        
        


       

    



       

