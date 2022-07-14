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

import torch
import poptorch
import popart
import numpy as np
import ctypes
import os


def constant_options(opts):
    
    opts._Popart.set("subgraphCopyingStrategy", int(popart.SubgraphCopyingStrategy.JustInTime))
    opts._Popart.set("outlineThreshold", 10.0)
    opts._Popart.set("accumulateOuterFragmentSettings.schedule",
                     int(popart.AccumulateOuterFragmentSchedule.OverlapMemoryOptimized))
    opts._Popart.set("accumulateOuterFragmentSettings.excludedVirtualGraphs", ["0"])
    opts.enableExecutableCaching('./cache_dir')

    engine_options = {
        "opt.useAutoloader": "true",
        "target.syncReplicasIndependently": "true",
    }

    opts._Popart.set("engineOptions", engine_options)

    mem_prop = {
        f'IPU0': .2 for i in range(1)
    }
    opts.setAvailableMemoryProportion(mem_prop)



def get_ipu_options():

    opts = poptorch.Options()
    constant_options(opts)
    
  
    opts.deviceIterations(1)
    # Precision options
    #if False:
    #opts.Precision.setPartialsType(torch.float16)



    return opts
