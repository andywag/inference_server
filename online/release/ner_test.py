#!/usr/bin/env python
import pika
import json



from ner_proto import NerProto
from bert_proto import BertInput
from dataclasses import asdict

connection = pika.BlockingConnection(pika.ConnectionParameters(host='192.168.3.114'))
channel = connection.channel()

channel.queue_declare(queue='ner')


result = channel.queue_declare(queue='', exclusive=True)
response_queue = result.method.queue

def on_response(self, ch, method, props, body):
    if self.corr_id == props.correlation_id:
        self.response = body

channel.basic_consume(
    queue=response_queue,
    on_message_callback=on_response,
    auto_ack=True)

    

def data(value):
    bert_input = BertInput([[value]*384], [[35]], [76])
    bert_input_dict = asdict(bert_input)
    bert_input_json = json.dumps(bert_input_dict)
    return bert_input_json

channel.basic_publish(exchange='', 
    routing_key='ner', 
    properties=pika.BasicProperties(
        reply_to=response_queue,
        correlation_id="100",
    ),
    body=data(100))


print(" [x] Sent 'Hello World!'")
connection.close()