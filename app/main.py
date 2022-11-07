from rabbitmq.connection.rabbitmq_producer import RabbitMQProducer
from rabbitmq.connection.rabbitmq_consumer import RabbitMQConsumer
from scdfutils import utils, ports
import pika
from datetime import datetime
import logging
import time
from scdfutils.http_status_server import HttpHealthServer
import os

HttpHealthServer.run_thread()


def on_send(self, _channel):
    """ Publishes data """
    logging.info("in on_send...")
    self.channel.basic_publish(self.exchange, self.routing_key, 'HELLO from ML Models',
                               pika.BasicProperties(content_type='text/plain',
                                                    delivery_mode=pika.DeliveryMode.Persistent,
                                                    timestamp=int(datetime.now().timestamp())))


def on_receive(self, header, body):
    logging.info(f"Received message...{body.decode('ascii')}")


def process():
    #######################################################
    # Producer code:
    #######################################################
    # Start publishing messages
    producer = ports.get_rabbitmq_port('producer',
                                       ports.FlowType.OUTBOUND,
                                       send_callback=on_send)

    time.sleep(5)

    # Can use to send more data
    producer.send_data('Hello again!')  # Can use to send more data

    time.sleep(5)

    #######################################################
    # Consumer code:
    #######################################################
    # Start consuming messages
    consumer = ports.get_rabbitmq_port('consumer',
                                       ports.FlowType.INBOUND,
                                       prefetch_count=0,
                                       receive_callback=on_receive)
