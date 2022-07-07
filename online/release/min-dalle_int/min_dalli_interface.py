# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import os
import pdb
import time
from datetime import datetime
import numpy as np
import poptorch

from min_dalle.min_dalle_torch import MinDalleTorch

class MinDalleInterfaceWrapper:
    def __init__(self):
        self.create_ipu()


    # TODO : Create multiple decoders per encoder for better performance
    def create_ipu(self):
        self.model = MinDalleTorch(False)
        print("Min Dalle Running")

    

    def run_data(self, input, callback):
        tic = time.time()
        
        text = input[0]
        seed = input[1]

        image = self.model.generate_image_serve(text, seed)
        callable(image)

       


def main():
    def callback(value):
        print("Here")
    text = """Avocado Chair"""

    wrapper = MinDalleInterfaceWrapper()
    result = wrapper.run_data((text, 10), callback)



if __name__ == '__main__':
    main()
