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

import ctypes
import datetime
import errno
import glob
import logging
import math
import os
from pathlib import Path
import random
import socket
import sys
import time
from collections import defaultdict
from functools import reduce
from itertools import chain

import numpy as np
import popart
import popdist
import popdist.popart

import utils
import utils.popvision as popvision
from bert_model import BertConfig, ExecutionMode, get_model
from bert_tf_loader import load_initializers_from_tf

from utils.device import acquire_device, device_is_replicated
from utils.distributed import popdist_root

from utils import packed_bert_utils

logger = logging.getLogger('BERT')




def set_library_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)


def bert_config_from_args(args):
    return BertConfig(**{k: getattr(args, k)
                         for k in BertConfig._fields if hasattr(args, k)})


def bert_add_inputs(args, model):
    config = bert_config_from_args(args)
    sequence_info = popart.TensorInfo("UINT32", [args.micro_batch_size * args.sequence_length])
    indices = model.builder.addInputTensor(sequence_info, "indices")
    positions = model.builder.addInputTensor(sequence_info, "positions")
    segments = model.builder.addInputTensor(sequence_info, "segments")

    masks = []
    mask_info = popart.TensorInfo("UINT32", [args.micro_batch_size, 1])
    masks.append(model.builder.addInputTensor(mask_info, "seq_pad_idx"))
        
    return indices, positions, segments, masks


def bert_logits_graph(model, indices, positions, segments, masks, mode):
    logits = model.build_graph(indices, positions, segments, masks)
    return logits



def bert_add_outputs(args, model, logits):
    if args.inference:
        outputs = bert_add_logit_outputs(model, logits)
        writer = None
    
    return outputs, writer


def bert_add_logit_outputs(model, logits):
    outputs = {}
    for logit in logits:
        outputs[logit] = popart.AnchorReturnType("ALL")
    for out in outputs.keys():
        model.builder.addOutputTensor(out)
    return outputs


def bert_session_options(args, model):
    engine_options = {}
    options = popart.SessionOptions()

    #TODO : Wait for Matmuloptions change to land
    options.matmulOptions['use128BitConvUnitLoad'] = 'true'
    options.matmulOptions['enableMultiStageReduce'] = 'false'
    options.matmulOptions['enableFastReduce'] = 'true'

    options.virtualGraphMode = popart.VirtualGraphMode.Manual
    options.enableFloatingPointChecks = args.floating_point_exceptions
    options.enableStochasticRounding = args.stochastic_rounding
    #options.enableGroupedMatmuls = False
    options.enablePrefetchDatastreams = not args.minimum_latency_inference
    options.enableOutlining = not args.no_outlining
    partials_type = "half" if args.enable_half_partials else "float"
    options.partialsTypeMatMuls = partials_type
    options.convolutionOptions = {'partialsType': partials_type}
    if args.replication_factor > 1:
        options.enableReplicatedGraphs = True
        options.replicatedGraphCount = args.replication_factor
        engine_options["target.syncReplicasIndependently"] = "true"
        engine_options["streamCallbacks.multiThreadMode"] = "collaborative"
        engine_options["streamCallbacks.numWorkerThreads"] = "auto"
        
    if args.use_popdist:
        popdist.popart.configureSessionOptions(options)
    # Increasing the outlineThreshold prevents creating subgraphs of cheap Ops
    # such as add or reshapeInplace.
    # Instead only reusing ops with a highSubgraphValue such as matmul or normalisation.
    options.outlineThreshold = 10.0
    if args.execution_mode == "PIPELINE":
        options.enablePipelining = True
        options.autoRecomputation = popart.RecomputationType.Pipeline
    

    if args.optimizer_state_offchip:
        options.optimizerStateTensorLocationSettings.location.storage = popart.TensorStorage.OffChip
    if args.replicated_tensor_sharding:
        options.optimizerStateTensorLocationSettings.location.replicatedTensorSharding = popart.ReplicatedTensorSharding.On
 
    if args.engine_cache is not None:
        build_path = os.getenv("INFERENCE_BUILD")
        if build_path is None:
            print("Please Set INFERENCE_BUILD environment variable")
            sys.exit(0)
        file_path = f"{build_path}/cache"
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        options.cachePath = f"{file_path}/{args.engine_cache}"
        options.enableEngineCaching = True
        #options.cachePath = args.engine_cache
    if args.profile:
        options.enableEngineCaching = False
    options.instrumentWithHardwareCycleCounter = args.report_hw_cycle_count
    options.disableGradAccumulationTensorStreams = not args.save_initializers_externally
    if args.max_copy_merge_size == -1:
        logger.debug("No copy merge size limit applied")
    else:
        logger.warning(
            f"Copy merge size limit set to {args.max_copy_merge_size}")
        engine_options["opt.maxCopyMergeSize"] = str(args.max_copy_merge_size)

    # Adding {"fullyConnectedPass", "TRAINING_BWD"} to some matmuls causes large
    # transposes before operations.
    if args.disable_fully_connected_pass:
        if args.task == "SQUAD" and args.sequence_length == 384:
            logger.warning(
                "Fully connected pass has been disabled. This may cause SQuAD 384 12-layer to go OOM.")
        options.enableFullyConnectedPass = False

    
    if args.variable_weights_inference:
        options.constantWeights = False

    if args.group_host_syncs:
        options.groupHostSync = True

    if args.internal_exchange_optimisation_target is not None:
        engine_options["opt.internalExchangeOptimisationTarget"] = str(args.internal_exchange_optimisation_target)

    options.engineOptions = engine_options

    # Set synthetic data mode (if active)
    if args.synthetic_data:
        if args.synthetic_data_initializer == "zeros":
            options.syntheticDataMode = popart.SyntheticDataMode.Zeros
        else:
            options.syntheticDataMode = popart.SyntheticDataMode.RandomNormal
        logger.info(
            f"Running with Synthetic Data Type '{options.syntheticDataMode}'")
    return options


def bert_session_patterns(args):
    patterns = popart.Patterns()
   
    if args.execution_mode == ExecutionMode.PIPELINE and args.recompute_checkpoint_every_layer and any(map(lambda l: l > 1, args.layers_per_ipu)):
        patterns.enablePattern("AccumulatePriorityPattern", True)

    
    return patterns


def compile_graph_checked(args, session):
    start_time = time.time()
    session.prepareDevice()
    end_time = time.time()
    compile_time = end_time - start_time
    logger.info(f"Compiled. Duration {compile_time} seconds")
    if args.profile:
        popvision.save_app_info({"compile_time": compile_time})
        if args.device_connection_type == "offline":
            sys.exit(0)


def save_model_and_stats(args, session, writer, step, epoch=None, step_in_filename=False):
    if True: #not args.no_model_save and popdist_root(args):
        save_file = "model"
        if epoch is not None:
            save_file += f"_{epoch}"
        if step_in_filename:
            save_file += f":{step}"

        if args.save_initializers_externally:
            save_dir = Path(args.checkpoint_dir, save_file)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = args.checkpoint_dir
        save_file += '.onnx'
        save_path = os.path.join(save_dir, save_file)
        save_vars = 'vars'.join(save_path.rsplit('model', 1))
        if args.save_initializers_externally:
            if hasattr(args, 'save_vars_prev') and os.path.exists(args.save_vars_prev):
                logger.debug(f'Updating external location for vars to {args.save_vars_prev}.')
                session.updateExternallySavedTensorLocations(args.save_vars_prev, save_vars)
        session.modelToHost(save_path)
        utils.save_model_statistics(save_path, writer, step)
        args.save_vars_prev = save_vars
        logger.info(f"Saved model to: {save_path}.")
        if args.save_initializers_externally:
            logger.info(f"Saved variables(weights and optimizer state) to: {save_vars}.")


def bert_writer(args):
    writer = None
    if args.log_dir is not None and popdist_root(args):
        log_name = f"{os.path.basename(args.checkpoint_dir)}."\
                   f"{datetime.datetime.now().isoformat()}"
        log_dir = os.path.join(
            args.log_dir, log_name)
        writer = SummaryWriter(log_dir=log_dir)
    return writer

def bert_inference_session(model, args, feed, device):
    options = bert_session_options(args, model)
    
    patterns = bert_session_patterns(args)

    proto = model.builder.getModelProto()

    #logger.info("Creating Session")
    
    session = popart.InferenceSession(fnModel=proto,
                                      deviceInfo=device,
                                      dataFlow=feed,
                                      patterns=patterns,
                                      userOptions=options)


    #logger.info("Compiling Inference Graph")
    compile_graph_checked(args, session)

    session.weightsFromHost()
    #session.setRandomSeed(args.seed)

    anchors = session.initAnchorArrays()

    return session, anchors


def bert_required_ipus(args, model):
    num_ipus = model.total_ipus
    num_ipus *= args.replication_factor
    return num_ipus


def bert_pretrained_initialisers(config, args):
    if args.synthetic_data:
        logger.info("Initialising from synthetic_data")
        return None

    if args.generated_data:
        logger.info("Initialising from generated_data")
        return None

    # The initialised weights will be broadcast after the session has been created
    if not popdist_root(args):
        return None

    init = None
    if args.onnx_checkpoint:
        logger.info(f"Initialising from ONNX checkpoint: {args.onnx_checkpoint}")
        init = utils.load_initializers_from_onnx(args.onnx_checkpoint, 
            limit_embedding_number=args.limit_embedding_number,
            limit_embedding_file=args.limit_embedding_file)

    if args.tf_checkpoint:
        logger.info(f"Initialising from TF checkpoint: {args.tf_checkpoint}")
        #print(f"Initialising from TF checkpoint: {args.tf_checkpoint}")
        init = load_initializers_from_tf(args.tf_checkpoint, True, config, args.task, inference=True)
    
    if args.torch_checkpoint_link:
        logger.info(f"Initialising from Torch checkpoint: {args.torch_checkpoint_name}")
        from bert_torch_loader import load_initializers_from_torch
        init = load_initializers_from_torch(args.torch_checkpoint_link, args.torch_checkpoint_name, config, args.task)

    return init



class BertRunner:
    def __init__(self, session, anchors, device):
        self.session = session
        self.anchors = anchors
        self.device = device

    def run(self, data):
        step = popart.PyStepIO(data, self.anchors)
        self.session.run(step)
        
        return self.anchors['Squad/Gemm:0']


    def stop(self,data):
        self.device.detach()


def main(args):
    so_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),"custom_ops.so")
    if os.path.exists(so_path):
        ctypes.cdll.LoadLibrary(so_path)
    else:
        logger.warning("Could not find custom_ops.so. Execute `make` before running this script.")

    set_library_seeds(args.seed)
    config = bert_config_from_args(args)
    initializers = bert_pretrained_initialisers(config, args)
    # Specifying ai.onnx opset9 for the slice syntax
    model = get_model(config,
                      mode=args.execution_mode,
                      initializers=initializers,
                      block=None)
    model.inference = True


    if config.use_packed_sequence_format:
        packed_bert_utils.add_inputs(model, True)
        logits = packed_bert_utils.logits_graph(model)
    else:
        indices, positions, segments, masks = bert_add_inputs(args, model)
        logits = bert_logits_graph(model, indices, positions, segments, masks, args.execution_mode)       

    outputs, writer = bert_add_outputs(args, model, logits)
    device = acquire_device(args, bert_required_ipus(args, model))
    data_flow = popart.DataFlow(args.batches_per_step, outputs)
    session, anchors = bert_inference_session(model, args, data_flow, device)
    #writer = bert_writer(args)
    if False:
        save_model_and_stats(args, session, writer, 0)

    runner = BertRunner(session, anchors, device)

    return runner, args


def create_model_external(config, override = dict()):
    args = utils.parse_bert_args(["--config", config])
    for key,value in override.items():
        args.__setattr__(key, value) 
    return main(args)

def create_args(config, override = dict()):
    args = utils.parse_bert_args(["--config", config])
    for key,value in override.items():
        args.__setattr__(key, value) 
    return args