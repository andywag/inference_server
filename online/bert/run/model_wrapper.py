import json
import random

# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import triton_python_backend_utils as pb_utils
import asyncio
import time
import sys
from popart_offline_triton_packed_async import BertInterfaceWrapper
import numpy as np
import queue
import zmq
import threading
import dill as pickle
import zlib
import traceback

class Temp:
    def __init__(self):
        pass

 
class ModelWrapper:
    def __init__(self, args):

        # Number of Possible Threads
        self.num_back = 2
        # Full Input Data Queue for BERT
        self.input_data_queue = queue.Queue(maxsize=1000000)
        # Full Output Data Queue for BERT
        self.output_data_queue = queue.Queue(maxsize=1000000)
        # Separate Output Queues
        self.output_queues = []
        self.length_queues = []
        self.final_queue = []
        for x in range(self.num_back):
            self.output_queues.append(queue.Queue()) 
            self.length_queues.append(queue.Queue()) 
            self.final_queue.append(queue.Queue()) 

        model_name = args['model_instance_name']
        model_indices = model_name.split("_")
        self.model_base, self.model_index = 0, 0
        
        if len(model_indices) == 3:
            self.model_base = int(model_indices[2]) % self.num_back
            self.model_index = int(int(model_indices[2])/self.num_back)
        print("Model Index : ", self.model_index, self.model_base)
        
        self.model_config = model_config = json.loads(args['model_config'])
        output0_config = pb_utils.get_output_config_by_name(model_config, "logits")
        self.output0_dtype = pb_utils.triton_string_to_numpy(output0_config['data_type'])
        print("Output Type", self.output0_dtype)

        output1_config = pb_utils.get_output_config_by_name(model_config, "query_ids_result")
        self.output1_dtype = pb_utils.triton_string_to_numpy(output1_config['data_type'])
        if self.model_base == 0:
            self._create_bert(args)
        
        # You must parse model_config. JSON string is not parsed here
        self.count = 0
  
        self.zmq_threads = []
        self.output_threads = []

        base_port = 50000 + self.model_index*(self.num_back - 1)
        if self.num_back > 1:
            if self.model_base == 0:
                for x in range(self.num_back-1):
                    worker_thread = threading.Thread(target = self._zmq_master_thread, args = (base_port + x, x + 1))
                    worker_thread.start()
                    self.zmq_threads.append(worker_thread)
                
                for x in range(self.num_back):
                    output_thread = threading.Thread(target = self._master_output_thread, args = (x,))
                    output_thread.start()
                    self.output_threads.append(output_thread)

                queue_splitter_thread = threading.Thread(target = self._queue_splitter, args = ())
                queue_splitter_thread.start()

            else:
                self.transmit_queue = queue.Queue()
                worker_thread = threading.Thread(target = self._zmq_worker_thread, args = (base_port + self.model_base-1,))
                worker_thread.start()
                self.zmq_threads.append(worker_thread)
       
    def _master_output_thread(self, index):
        while True:
            transmit_length = self.length_queues[index].get()
            logits_response = -10*np.ones((transmit_length,2*self.bert.internal_args.sequence_length),dtype=np.float32)
            id_group = []
            #print(f"Waiting in Queue {index} for {transmit_length}")
            for x in range(transmit_length):
                _, ids, result = self.output_queues[index].get()
                if x % 2000 == 1999:
                    print("Getting Data", index, x, transmit_length, self.model_base)
                id_group.append(ids)
                logits_response[x,:] = result
            self.final_queue[index].put((id_group, logits_response))
            print("Returning", self.model_base, len(id_group))
            #print("Added Data to Final Queue", index, self.model_base, self.final_queue[index].qsize())
 

    def _queue_splitter(self):
        while True:
            data = self.output_data_queue.get()
            if data[0] < self.num_back:
                self.output_queues[data[0]].put(data)
            else:
                print("Can't Write to Queue", data[0], data)

    def _zmq_master_thread(self, port, index):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % port)
        while True:
            try :
                input_ids = socket.recv()
                segment_ids = socket.recv()
                query_ids = socket.recv()
                 
                input_ids = np.frombuffer(input_ids, dtype=np.int32)
                input_ids = input_ids.reshape( (int(len(input_ids)/384), 384))
                segment_ids = np.frombuffer(segment_ids, dtype=np.int32).reshape(len(input_ids), 1)
                query_ids = np.frombuffer(query_ids, dtype=np.uint64)
                #print("Finished Loading Data", len(input_ids))

                self.length_queues[index].put(len(input_ids))
                for x in range(len(input_ids)):
                    self.input_data_queue.put((input_ids[x], segment_ids[x], query_ids[x], index))

                id_group, logits_response = self.final_queue[index].get()
                socket.send(np.asarray(id_group,dtype=np.uint64), zmq.SNDMORE)
                socket.send(logits_response)


                #time.sleep(30)
                #socket.send_string("Done")

            except Exception as e:
                print(e)
                traceback.print_exc()
                time.sleep(3)


    def _zmq_worker_thread(self, port):
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        socket.connect ("tcp://localhost:%s" % port)
        while True:
            try:
                request = self.transmit_queue.get()
                socket.send(pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy(),zmq.SNDMORE)
                socket.send(pb_utils.get_input_tensor_by_name(request, "segment_ids").as_numpy(),zmq.SNDMORE)
                socket.send(pb_utils.get_input_tensor_by_name(request, "query_ids").as_numpy())

                id_group = socket.recv()
                logits = socket.recv()

                id_group = np.frombuffer(id_group, dtype=np.uint64)
                logits = np.frombuffer(logits, dtype=np.float32)
                logits = logits.reshape(int(len(logits)/768),768)

                self.final_queue[0].put((id_group,logits))

            except Exception as e:
                print(e)
                traceback.print_exc()


    def _create_bert(self, args):
  
        build_path = "/localdata/andyw/projects/public_examples/applications/popart/bert_inference/run/build"
        print("Input Config", args)

        args = Temp()
        args.task = "Squad"
        args.build_path = build_path
        args.max_examples = 0
        args.features_cache_path = f"{build_path}/eval_features.pickle"
        args.vocab_file_path = f"{build_path}/data/bert_tf_v1_1_large_fp32_384_v2/vocab.txt"
        args.orig_squad_path = f"{build_path}/data/dev-v1.1.json"
        args.accuracy_predictions_path = f"{build_path}/result/predictions.json"

        sys.argv  = ["","","",""]
        import tensorflow

        print("Config", args)


        self.bert = BertInterfaceWrapper(args, 
            config="/localdata/andyw/projects/public_examples/applications/popart/bert_inference/configs/sut_inference_pack_384_triton.json",
            input_data_queue=self.input_data_queue,
            output_data_queue =self.output_data_queue)



    def get_final_queue(self, index):
        #print("Waiting for Final Queue to Fill", self.model_base)
        id_group, logits_response = self.final_queue[0].get()
        print("Output Final Data", len(id_group), self.model_base)

        logits = pb_utils.Tensor("logits", logits_response.astype(self.output0_dtype))
        query_return_np = np.asarray(id_group, dtype=np.uint64).astype(self.output1_dtype)
        #print("Query Return",logits_response.shape)
        query_ids_result = pb_utils.Tensor("query_ids_result", query_return_np)
        
        inference_response = pb_utils.InferenceResponse(output_tensors=[query_ids_result, logits])
        return [inference_response]


    def _execute_local(self, requests):
        #self.input_data_queue.put(requests)
        if len(requests) > 1 :
            print("Failed Executtion")

        request = requests[0]

        input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids").as_numpy()
        data_length = len(input_ids)

        segment_ids = pb_utils.get_input_tensor_by_name(request, "segment_ids").as_numpy()
        query_ids = pb_utils.get_input_tensor_by_name(request, "query_ids").as_numpy().flatten().tolist()
        
        self.length_queues[0].put(data_length)
        #self.input_queue_count += 1
        #print("Putting Data Local", data_length, self.input_queue_count)
        for x in range(data_length):
            self.input_data_queue.put((input_ids[x], segment_ids[x], query_ids[x], 0))

        return self.get_final_queue(0)


    def execute(self, requests):
        #print("Executing Request", self.model_base)
        if self.model_base == 0:
            result = self._execute_local(requests)
            print("Returning Result", self.model_base)
            return result
        else:
            self.transmit_queue.put(requests[0])
            result = self.get_final_queue(0)
            print("Returning Result", self.model_base)
            return result
