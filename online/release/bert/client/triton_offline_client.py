# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from tritonclient.utils import *
import tritonclient.grpc as grpcclient
import argparse
import subprocess

import sys
import os
sys.path.append("../run")

import numpy as np
import time
import mlperf_loadgen as lg
from   squad_QSL import get_squad_QSL
import math
import numpy as np
import array
import threading
import queue

from client_base import SUTBase, run_loadgen, get_args, first_non_zero

import logging
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.ERROR)

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

model_name = "squad"
#model_name = "bert"

class SUT(SUTBase):
    def __init__(self, qsl, lg, number_threads = 2, groups = 128, number_backends = 4):
        super().__init__(qsl,lg,number_threads, groups)
    
        worker_threads = []
        for x in range(self.number_threads):
            worker_threads.append(threading.Thread(target = self.send_thread, args = ()))
            worker_threads[x].start()
        
    def send_thread(self):
        while True:
            data_thread = self.input_queue.get()
            self.single_packet(data_thread)


    def issue_queries(self, query_samples):
        self.total_count = len(query_samples)
        group = self.groups
        iterations = int(math.ceil(len(query_samples)/group))
        for x in range(iterations):
            sp = x*group
            ep = min((x+1)*group, len(query_samples)) 
            self.input_queue.put(query_samples[sp:ep])   
              
        time.sleep(2)
            
def main():
    args = get_args()
    run_loadgen(args, SUT)

if __name__ == "__main__":
    main()
