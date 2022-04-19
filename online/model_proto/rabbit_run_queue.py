import pika
import json

from dataclasses import asdict
import threading 
import time

class RabbitRunQueue:

    def __init__(self, queue_name:str, host:str):
        self.queue_name = queue_name

        params = pika.ConnectionParameters(heartbeat=30, host=host)
        # Transmit Channel
        tx_connection = pika.BlockingConnection(params)
        self.tx_channel = tx_connection.channel()
        self.tx_channel.queue_declare(queue=queue_name)

        self.query_dict = dict()

        # Receive Channel
        rx_connection = pika.BlockingConnection(params)
        self.rx_channel = rx_connection.channel()
        result = self.rx_channel.queue_declare(queue='', exclusive=True)


        self.response_queue = result.method.queue

        run = threading.Thread(target=self._run)
        run.start()

        heartbeat = threading.Thread(target=self._heartbeat)
        heartbeat.start()

    def _heartbeat(self):
        while True:
            time.sleep(10)
            self.tx_channel.basic_publish(exchange='', 
                routing_key=self.queue_name, 
                properties=pika.BasicProperties(
                    reply_to=self.response_queue,
                    type='my_heartbeat'
                ),
                body = "")

    def _run(self):
        
        # Start the receive Channel Consuming
        self.rx_channel.basic_consume(
            queue=self.response_queue,
            on_message_callback=self.on_response,
            auto_ack=True)

        self.rx_channel.basic_qos(prefetch_count=100); 
        self.rx_channel.start_consuming()
        while True:
            time.sleep(10)
        

    def on_response(self, ch, method, props, body):
        if not props.type == 'my_heartbeat':
            print("R", props.correlation_id, self.query_dict[props.correlation_id])
            self.query_dict[props.correlation_id](body)
            del self.query_dict[props.correlation_id]

    def post_message(self, message_body:json, query_id:str, callback_fn):
        print("Posting Message")
        self.query_dict[query_id] = callback_fn
        self.tx_channel.basic_publish(exchange='', 
            routing_key=self.queue_name, 
            properties=pika.BasicProperties(
                reply_to=self.response_queue,
                correlation_id=query_id,
            ),
            body=message_body)

