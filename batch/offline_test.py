from offline.infer import main
from offline.offline_config import InferDescription

import logging
logger = logging.getLogger()

import pickle

fv = pickle.load( open( "save.pik", "rb" ) )
print("Size", len(fv))
print("S", len(fv[0]))
for x in range(len(fv)):
    result = fv[x]
    #print("AAA", result[0])
    for y in range(len(result)):
        #print("Result", result[y])
        pass
        #logger.info(f"B {result[1]} {result[1].shape} {y} {result[1][y]}")
        #index = result[1][y].item()
        #print("A", result[0].shape, result[0][y][index].item())
        #results.append({'class':index,'probability':result[0][y][index].item()})


print(fv[0][1].shape)


#from datasets import load_dataset
#dataset = load_dataset("indonlu","emot")
#dataset = dataset['train']
#print(dataset[2])
#print(len(dataset))

#wikitext:wikitext-103-v1,test,text

#dataset = load_dataset("wikitext","wikitext-103-v1")
#dataset = dataset['test']
#print(dataset.column_names)

