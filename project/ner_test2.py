#!/usr/bin/env python
import pika
import uuid
import time
import sys
sys.path.append('../model_proto')
from rabbit_run_queue import RabbitRunQueue

from bert_proto import BertInput
from dataclasses import asdict
import json

bert_input = BertInput([[52]*384], [[35]], [76])
bert_input_dict = asdict(bert_input)
bert_input_json = json.dumps(bert_input_dict)

def data(value):
    bert_input = BertInput([[value]*384], [[35]], [76])
    bert_input_dict = asdict(bert_input)
    bert_input_json = json.dumps(bert_input_dict)
    return bert_input_json

run_queue = RabbitRunQueue('ner')

def callback(body):
    print("Received")

for x in range(10):
    run_queue.post_message(data(x+1), str(x), callback)
    time.sleep(1)
