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


def training_options(opts, ipu_options):
    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    opts.Precision.enableStochasticRounding(True)
    opts.autoRoundNumIPUs(True)
    #opts.anchorMode(poptorch.AnchorMode.Sum)
    # PopART options
    opts._Popart.set("disableGradAccumulationTensorStreams", True)


    opts.randomSeed(55)

    opts.Training.gradientAccumulation(ipu_options.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    opts.Training.setAutomaticLossScaling(ipu_options.auto_loss_scaling)

    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        .useOnChipStorage(not ipu_options.optimizer_state_offchip)
        .useReplicatedTensorSharding(ipu_options.replicated_tensor_sharding)
    )

    mem_prop = {
        f'IPU{i}': ipu_options.matmul_proportion[i]
        for i in range(int(ipu_options.ipus_per_replica))
    }
    opts.setAvailableMemoryProportion(mem_prop)

def get_options(options, train:bool=False):

    opts = poptorch.Options()
    constant_options(opts)
    
    if train:
        training_options(opts, options)

    
    opts.deviceIterations(options.batches_per_step)
    # Precision options
    if options.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)



    return opts
