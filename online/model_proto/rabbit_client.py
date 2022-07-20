from dataclasses import dataclass, asdict
import pika, sys, os
import json
import numpy as np
from functools import partial

class RabbitProtoWrapper:
    def __init__(self, model_proto, config):
        self.model_proto = model_proto

        # Connect to the rabbit queue
        params = pika.ConnectionParameters(heartbeat=30, host=config['rabbit']['host'])
        # Receive Channel
        connection = pika.BlockingConnection(params)
        self.rx_channel = connection.channel()
        self.rx_channel.queue_declare(queue=model_proto.name)
        # Transmit Channel
        connection = pika.BlockingConnection(params)
        self.tx_channel = connection.channel()



    def finish_callback(self, reply_to, correlation_id, result):
        """ Handle the callback from the model """
        if len(result) == 2:
            output = self.model_proto.output_type.create(result[1])
        else:
            output = self.model_proto.output_type.create(result)
        output_dict = asdict(output)
        output_json = json.dumps(output_dict)

        #print("Finish Callback", reply_to, correlation_id)
        self.tx_channel.basic_publish(exchange='',
            routing_key=reply_to,
            properties=pika.BasicProperties(
                correlation_id=correlation_id,
            ),
            body=output_json)

    def send_callback(self, ch, method, properties, body):
        # Handle Heartbeat : Rabbit connections need to be kept alive when not operating
        if properties.type == 'my_heartbeat':
            self.tx_channel.basic_publish(exchange='',
                routing_key=properties.reply_to,
                properties=pika.BasicProperties(
                    type='my_heartbeat',
                ),
                body = "")
        elif len(body) != 0: # Handle message sent to the model
            data = json.loads(body) 
            # Convert the input to the model based input
            input_data = self.model_proto.body_to_input(data)
            input_list = input_data.items()
            # Create the callback to handle the model queury
            callback = partial(self.finish_callback, properties.reply_to, properties.correlation_id)
            # Run the model 
            self.model_proto.model.run_data(input_list, callback)
        

    def run(self):
        # Setup the Receive Channel
        self.rx_channel.basic_qos(prefetch_count=5); 
        self.rx_channel.basic_consume(queue=self.model_proto.name, on_message_callback=self.send_callback, auto_ack=True)
        self.rx_channel.start_consuming()
