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
import popdist
import popdist.poptorch
import numpy as np
import ctypes
import os

from .offline_config import Ipu

def constant_options(opts):
        # PopART options
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


def get_options(options:Ipu):


    opts = poptorch.Options()
    constant_options(opts)
    
    opts.deviceIterations(options.batches_per_step)

    #mem_prop = {
    #    f'IPU{i}': model_description.ipu_layout.matmul_proportion[i]
    #    for i in range(int(model_description.ipu_layout.ipus_per_replica))
    #}
    #opts.setAvailableMemoryProportion(mem_prop)

    '''
    Set ipu specific options for the model, see documentation:
    https://docs.graphcore.ai/en/latest/
    '''



    # Numpy options



    


    #if config.executable_cache_dir:
    #    opts.enableExecutableCaching(config.executable_cache_dir)

    # Precision options
    if options.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)



    return opts
