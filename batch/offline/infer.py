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

from .infer_classes import Base, Sequence, Token, MLM


def create_dataset(dataset, model_class:Base, options):
    """ Function to create a dataset loader. Assumes a hugging face dataset with column containing [text, optional(label)] """
    inference_config = model_class.inference_config
    tokenizer = AutoTokenizer.from_pretrained(inference_config.tokenizer, use_fast=True)
    tokenized_dataset = dataset.map(lambda x: tokenizer(x['text'],
        max_length=inference_config.detail.sequence_length, truncation=True, pad_to_max_length=True),batched=True)

    columns = model_class.dataset_columns
    if 'label' in dataset.column_names:
        columns.append('label')
    tokenized_dataset.set_format(type='torch', columns=columns)

    data_loader = poptorch.DataLoader(options, tokenized_dataset, batch_size=inference_config.detail.batch_size, shuffle=False)

    return data_loader

def decode_dataset_tag(tag):
    data_tag = tag.split(",")

    data_internal = data_tag[0].split(":")
    if len(data_internal) == 1:
        dataset = load_dataset(data_internal[0])
    else:
        dataset = load_dataset(data_internal[0], data_internal[1])
    for tag in data_tag[1:-1]:
        dataset = dataset[tag]
    
    if data_tag[-1] != 'text':
        dataset = dataset.map(lambda x:{'text':data_tag[-1]})

    return dataset

def create_data_loader(model_class:Base, options):
    dataset = decode_dataset_tag(model_class.inference_config.dataset)
    return create_dataset(dataset, model_class, options)



def handle_error(message:str, e=None):
    logger.error(message)
    if e is not None:
        traceback.print_exc()
    sys.exit(1)

def update_status(mongo, t, m=None):
    if mongo is not None:
        mongo.update_status(t,m)

def main(inference_config:InferDescription, mongo, celery, logger):
    
    if inference_config.classifier.classifier_type == 'Sequence':
        model_class = Sequence(inference_config)
    elif inference_config.classifier.classifier_type == 'Token':
        model_class = Token(inference_config)
    elif inference_config.classifier.classifier_type == 'MLM':
        model_class = MLM(inference_config)
    else:
        handle_error("Classifier Not Found") 

    options = get_options(inference_config.ipu)

    # Handle Data Loading
    update_status(mongo, "Data")
    try :    
        data_loader = create_data_loader(model_class, options)
    except Exception as e:
        update_status(mongo, "DataError",str(e))
        handle_error(f"Data Loading Error {str(e)}",e)
        return

    # Model Compilation
    update_status(mongo, "Model")
    try :
        config = model_class.create_config()
        model = model_class.model.from_pretrained(model_class.inference_config.checkpoint, config=config).half()
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
        model_ipu.compile(*model_class.compile_inputs(first_data))

    except Exception as e:
        update_status(mongo, "CompileError",str(e))
        handle_error(e)
        return 

    update_status(mongo, "Running")

    errors,samples = 0,0
    
    start_time = time.time()
    while True:
        
        if first_data is not None:
            data = first_data
            first_data = None
        else:
            try :
                data = next(iter_loader)
            except Exception as e:
                logger.info(f"Finished with Dataset {e}")
                break

        try :
            result = model_ipu(*model_class.model_inputs(data))
        except Exception as e:
            update_status(mongo, "RunError",str(e))
            handle_error("Run Error", e)
            return

        samples += inference_config.detail.batch_size*inference_config.ipu.batches_per_step 
        
        error = model_class.handle_result(result, data)
            
        if error is not None:
            errors += error
            logger.info(f"Accuracy : {errors/samples} QPS : {samples/(time.time()-start_time)} Errors : {error}")
       
            if mongo is not None:
                mongo.update_accuracy(accuracy=1.0-errors/samples,qps=samples/(time.time()-start_time))
        else:
            
            logger.info(f"QPS : {samples/(time.time()-start_time)} , Time : {(time.time()-start_time)}")
            mongo.update_accuracy(accuracy=0.0,qps=samples/(time.time()-start_time))
    update_status(mongo, "Finished")

    model_ipu.detachFromDevice()
    model_class.post_process(mongo)


    return {"status":"Success", "results":[]}

if __name__ == "__main__":
    inference_config = InferDescription()
    main(inference_config)