from transformers import AutoTokenizer, AutoConfig
from .ipu_options import get_options
from datasets import load_dataset

import poptorch

from .offline_config import InferDescription
import sys
import logging
logger = logging.getLogger()

import traceback
import numpy as np
import time
import torch
from adlfs import AzureBlobFileSystem

from .infer_classes import Base, Sequence, Token, MLM
from datasets import load_from_disk
import sys

from cloud_utils import CloudFileContainer
from .optimization import get_optimizer

def handle_custom_ops(config):
    file_dir = os.path.dirname(os.path.realpath(__file__))
    CUSTOM_OP_PATH = os.path.join(file_dir, "custom_ops.so")
    if os.path.exists(CUSTOM_OP_PATH):
        ops_and_patterns = ctypes.cdll.LoadLibrary(CUSTOM_OP_PATH)
        ops_and_patterns.setVocabSize(config.vocab_size)
        ops_and_patterns.setEmbeddingSize(config.hidden_size)
        ops_and_patterns.setHiddenSize(config.hidden_size)
    else:
        exit()

def create_dataset(dataset, model_class:Base, options, train:bool=False):
    """ Function to create a dataset loader. Assumes a hugging face dataset with column containing [text, optional(label)] """
    inference_config = model_class.inference_config
    tokenizer = AutoTokenizer.from_pretrained(inference_config.tokenizer, use_fast=True)
    # TODO : Offset mapping only used for offset_mapping in NER
    #tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
    #    max_length=inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True,return_offsets_mapping=True),batched=True)

    tokenized_dataset = model_class.tokenize(tokenizer, dataset)
    columns = model_class.dataset_columns
    if 'label' in dataset.column_names:
        columns.append('label')
        
    tokenized_dataset.set_format(type='torch', columns=columns)

    shuffle = False
    if train:
        shuffle = True
    data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=inference_config.detail.batch_size, shuffle=shuffle)

    return data_loader

def decode_dataset_tag(tag, cloud_file_system):
    return cloud_file_system.load_dataset(tag)


def create_data_loader(model_class:Base, options, cloud_file_system, train:bool=False):
    dataset = decode_dataset_tag(model_class.inference_config.dataset, cloud_file_system)
    return create_dataset(dataset, model_class, options, train)



def handle_error(message:str, e=None):
    logger.error(message)
    if e is not None:
        traceback.print_exc()
    sys.exit(1)

def update_status(mongo, t, m=None):
    if mongo is not None:
        mongo.update_status(t,m)

def main(inference_config:InferDescription, train:bool, mongo, celery, logger):
    print(f"A {inference_config}")

    cloud_file_system = CloudFileContainer(inference_config.cloud, inference_config.endpoint)
    logger.info(f"File System {cloud_file_system}")

    if train:
        inference_config.detail.batch_size=2

   
    if inference_config.classifier.classifier_type == 'Sequence':
        model_class = Sequence(inference_config)
    elif inference_config.classifier.classifier_type == 'Token':
        model_class = Token(inference_config)
    elif inference_config.classifier.classifier_type == 'MLM':
        model_class = MLM(inference_config)
    else:
        handle_error("Classifier Not Found") 

    options = get_options(inference_config.ipu, train)

    # Handle Data Loading
    update_status(mongo, "Data")
    try :    
        data_loader = create_data_loader(model_class, options, cloud_file_system, train)
    except Exception as e:
        update_status(mongo, "DataError",str(e))
        handle_error(f"Data Loading Error {str(e)}",e)
        return

    # Model Compilation
    update_status(mongo, "Model")
    try :
        config = model_class.create_config(train)
        model = model_class.model.from_pretrained(model_class.inference_config.checkpoint, config=config).half()
        if train:
            optimizer = get_optimizer(inference_config.optimizer, model)
            model.train()
            model_ipu = poptorch.trainingModel(model, options, optimizer)
        else:
            model_ipu = poptorch.inferenceModel(model, options)

    except Exception as e:
        update_status(mongo, "ModelError",str(e))
        handle_error(f"Data Loading Error {str(e)}",e)
        return 

    iter_loader = iter(data_loader)
    update_status(mongo, "Compiling")
    tic = time.time()
    try :
        first_data = next(iter_loader)
        #print("A", first_data['input_ids'].dtype, first_data['input_ids'].shape)
        #input_data = model_class.compile_inputs(first_data)
        
        model_ipu.compile(*model_class.model_inputs(first_data))

    except Exception as e:
        logger.info(f"A{e}")
        traceback.print_exc()
        update_status(mongo, "CompileError",str(e))
        handle_error(e)
        return 

    update_status(mongo, "Running")

    errors,samples = 0,0
    epoch = 0

    start_time = time.time()
    while True:
        
        if first_data is not None:
            data = first_data
            first_data = None
        else:
            try :
                data = next(iter_loader)
            except Exception as e:
                if train: 
                    if inference_config.optimizer.epochs is not None and epoch == inference_config.optimizer.epochs-1:
                        break
                    iter_loader = iter(data_loader)
                    data = next(iter_loader)  
                    epoch += 1
                else:
                    logger.info(f"Finished with Dataset {e}")
                    break

        try :
            result = model_ipu(*model_class.model_inputs(data))
            #logger.info(f"{len(result)}")
            #logger.info(f"{result[0].shape}")
            #logger.info(f"{result[1].shape}")

        except Exception as e:
            update_status(mongo, "RunError",str(e))
            handle_error("Run Error", e)
            return

        samples += inference_config.detail.batch_size*inference_config.ipu.batches_per_step *inference_config.ipu.gradient_accumulation
        
        error = model_class.handle_result(result, data)
            
        if error is not None:
            errors += error
            logger.info(f"Accuracy : {errors/samples} QPS : {samples/(time.time()-start_time)} Errors : {error}")
       
            if mongo is not None:
                mongo.update_accuracy(accuracy=1.0-errors/samples,qps=samples/(time.time()-start_time))
        else:
            if train:
                logger.info(f"Loss : {result[0].item()} QPS : {samples/(time.time()-start_time)}  Time : {(time.time()-start_time)}")
                if mongo is not None:
                    mongo.update_loss(loss=result[0].item(),qps=samples/(time.time()-start_time))
            else:
                logger.info(f"QPS : {samples/(time.time()-start_time)} , Time : {(time.time()-start_time)}")
                mongo.update_accuracy(accuracy=0.0,qps=samples/(time.time()-start_time))
    update_status(mongo, "Finished")

    model_ipu.detachFromDevice()
    model_class.post_process(mongo, cloud_file_system)


    return {"status":"Success", "results":[]}

if __name__ == "__main__":
    inference_config = InferDescription()
    main(inference_config)