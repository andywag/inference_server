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

from client_base import SUTBase, run_loadgen, get_args, first_non_zero, main

import logging
log = logging.getLogger()
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.DEBUG)

model_name = "bert"
shape = [4]


class SUT(SUTBase):
    def __init__(self, qsl, lg, number_threads = 2, groups = 32, number_backends = 4):
        super().__init__(qsl,lg,number_threads,groups)
        self.input_list = []

    def callback(self, result, error):
        if error is not None:
            log.error(f"Error {error}")
        response = result
        logits = response.as_numpy("logits")
        query_ids = response.as_numpy("query_ids_result")
        self.total_rx_query += len(query_ids)

        try:
            for x in range(len(query_ids)):
                response_array = array.array("B", logits[x,:].tobytes())   # this is a bytes' array
                bi = response_array.buffer_info()
                response = self.lg.QuerySampleResponse(query_ids[x], bi[0], bi[1])
                self.lg.QuerySamplesComplete([response])
        except Exception as e:
            import traceback
            print("Error", e)
            traceback.print_exc()

    def issue_queries(self, query_samples):
        self.input_list.append(query_samples[0])
        if len(self.input_list) == self.groups:
            self.single_packet(self.input_list, self.callback)
            self.input_list = []
                       
       
    def flush_queries(self):
        self.single_packet(self.input_list, self.callback)
        print("Flushing Q", self.total_rx_query, self.total_tx_query)
        time.sleep(2)
        #self.issue_query_single(self.last_id, self.last_feature, True)
        #print("flush queries")

 
if __name__ == "__main__":
    main(SUT, server = True)
