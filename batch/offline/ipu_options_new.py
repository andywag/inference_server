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


def constant_options(opts):
    opts.setExecutionStrategy(poptorch.PipelinedExecution(poptorch.AutoStage.AutoIncrement))
    opts.Precision.enableStochasticRounding(True)
    opts.autoRoundNumIPUs(True)
    opts.anchorMode(poptorch.AnchorMode.Sum)
        # PopART options
    opts._Popart.set("disableGradAccumulationTensorStreams", True)
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

def training_options(opts, model_description):
    opts.randomSeed(model_description.execution_description.random_seed)

    opts.deviceIterations(model_description.execution_description.batches_per_step)
    opts.Training.gradientAccumulation(model_description.execution_description.gradient_accumulation)
    opts.Training.accumulationAndReplicationReductionType(poptorch.ReductionType.Mean)
    opts.Training.setAutomaticLossScaling(model_description.ipu_options.auto_loss_scaling)

    opts.TensorLocations.setOptimizerLocation(
        poptorch.TensorLocationSettings()
        .useOnChipStorage(not model_description.ipu_options.optimizer_state_offchip)
        .useReplicatedTensorSharding(model_description.ipu_options.replicated_tensor_sharding)
    )

    mem_prop = {
        f'IPU{i}': model_description.ipu_layout.matmul_proportion[i]
        for i in range(int(model_description.ipu_layout.ipus_per_replica))
    }
    opts.setAvailableMemoryProportion(mem_prop)


def get_options(options):

    opts = poptorch.Options()
    constant_options(opts)
    
    opts.deviceIterations(options.batches_per_step)
    # Precision options
    if options.enable_half_partials:
        opts.Precision.setPartialsType(torch.float16)

    if options.training:
        training_options(opts,options)


    return opts
