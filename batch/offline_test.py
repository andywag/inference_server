from offline.infer import main
from offline.offline_config import InferDescription

import logging
logger = logging.getLogger()

from datasets import load_dataset
dataset = load_dataset("indonlu","emot")
dataset = dataset['train']
print(dataset[2])
print(len(dataset))

#infer_dataset = inference_config.dataset
#    data_tag = infer_dataset.split(":")
#    dataset = load_dataset(data_tag[0])
#    print("Starting", data_tag)
#    for tag in data_tag[1:-1]:
#        dataset = dataset[tag]
#    print("Loading DataSet")


#if __name__ == "__main__":
#    inference_config = InferDescription()
#    main(inference_config,None,None,logger)