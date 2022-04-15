import sys
import os
import logging
import argparse
import time
import logging
#from bert_wrapper_zmq import BertWrapper
import multiprocessing as mp

from general_wrapper import GeneralWrapper
from bert_serve_model import BertServeModel

log = logging.getLogger()
logging.basicConfig(format='[%(filename)s:%(lineno)d] %(message)s', level=logging.DEBUG)

model_name = "bert"
shape = [4]

def get_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--instances", type=int, default=1, help="Number of Instances Per Model")
    parser.add_argument("--port", type=int, default=50000, help="Initial Port to Use")
    parser.add_argument("--backends", type=int, default=4, help="Number of Zmq Backends")

    args = parser.parse_args()

    return args


#def create_bert(args, port, ner, backends = 4):
#    wrapper = BertWrapper(args, port, backends, ner)
#    while True:
#        time.sleep(10)
    
def create_bert(args, port, ner):
    model = BertServeModel(ner)
    GeneralWrapper(model, port, num_backends=args.backends)
    while True:
        time.sleep(10)


def main():
    args = get_args()

    bert_process = mp.Process(target=create_bert, args=(args,args.port,False))
    ner_process = mp.Process(target=create_bert, args=(args,args.port+16,True))
    
    bert_process.start()
    ner_process.start()

    while True:
        time.sleep(10)


if __name__ == "__main__":
    main()
