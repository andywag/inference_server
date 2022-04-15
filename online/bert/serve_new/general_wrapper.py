# triton_python_backend_utils is available in every Triton Python model. You
# need to use this module to create inference requests and responses. It also
# contains some utility functions for extracting information from model_config
# and converting Triton input/output types to numpy types.
import time
import sys

#from popart_offline_triton import BertInterfaceWrapper
import sys

from bert_interface_wrapper import BertInterfaceWrapper

import numpy as np
import queue
import zmq
import threading
import traceback


import logging
log = logging.getLogger()
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.INFO)




class GeneralWrapper:
    def __init__(self, model, port = 50000, num_backends=1):
        self.model = model
        self.result_queue = queue.Queue()
        

        self.zmq_threads = []

        for x in range(num_backends):
            worker_thread = threading.Thread(target = self._zmq_master_thread, args = (port + x, x))
            worker_thread.start()
            self.zmq_threads.append(worker_thread)

    def _zmq_master_thread(self, port, index):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind("tcp://*:%s" % port)
        while True:
            try :
                input_data = self.model.handle_zmq_receive(socket)
                output_data = self.model.run(input_data)
                self.model.handle_zmq_transmit(socket, output_data)
                
            except Exception as e:
                print(e)
                traceback.print_exc()
                time.sleep(3)


  
       
     









    



