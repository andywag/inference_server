import requests
import time

import graphcore_api as gr
import threading
import numpy as np
import argparse
from multiprocessing import Pool
from api_classes import *


basic_question_list = ["Where do I live?","My name is Wolfgang and I live in Berlin"]
basic_ner_list = ["Where do I live? My name is Wolfgang and I live in Berlin"]

SQUAD = 'Squad'
NER = 'Ner'

def single_run(model = SQUAD, group_size=None):
    if model == SQUAD:
        if group_size is None:
            return Squad("Where do I live?","My name is Wolfgang and I live in Berlin")
        else:
            return SquadArray([basic_question_list]*group_size)
    elif model == NER:
        if group_size is None:
            return Ner("Where do I live? My name is Wolfgang and I live in Berlin")
        else:
            return NerArray(basic_ner_list*group_size)
    else:
        print("Model Not Found")

def serial_run(num, model=SQUAD, group_size=None):
    times = [0]*num
    tic = time.time()
    for x in range(num):
        gr.post(single_run(model, group_size))
        times[x] = time.time()
    if group_size is None:
        print(f"Serial {model} : {num} Runs : {1000*np.mean(np.diff(times))} ms Average Latency")
    else:
        print(f"Serial {model} : {num} Runs : {group_size} Array : {1000*np.mean(np.diff(times))} ms Average Latency")
 

def thread_run(num, parallel=4, model=SQUAD, group_size=None):
    for i in range(parallel):
        thr = threading.Thread(target=serial_run, args=(num, model, group_size))
        thr.start()
    


def get_args():
    parser = argparse.ArgumentParser()   
    parser.add_argument("--group", type=int, default=1, help="Size of Groups")
    parser.add_argument("--parallel", type=int, default=1, help="Number of Parallel Jobs")
    parser.add_argument("--number", type=int, default=50, help="Length of Run")
    parser.add_argument("--job", type=str, default='squad', help="Job Type")

    args = parser.parse_args()

    return args

def main():
    serial_run(10)
    serial_run(10, model=NER)

    serial_run(10, group_size=20)
    serial_run(10, model=NER, group_size=20)

    thread_run(10)
    thread_run(10, model=NER)

    #squad_run(5)
    #ner_run(5,10)
    #squad_array_run(5, 2)
    #ner_array_run(5,2)

    #args = get_args()
    #if args.group == 1:
    #    if args.job == 'squad' or args.job == 'all':
    #        parallel_thread_squad(args.number, args.parallel)
    #    if args.job == 'ner' or args.job == 'all':
    #        parallel_thread_ner(args.number, args.parallel)
    #else:
    #    if args.job == 'squad' or args.job == 'all':
    #        parallel_thread_squad_array(args.number, args.parallel)
    #    if args.job == 'ner' or args.job == 'all':
    #        parallel_thread_ner(args.number, args.parallel)


if __name__ == "__main__":
    main()


