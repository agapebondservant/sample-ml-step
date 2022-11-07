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
logger = logging.getLogger('mlmodeltest')


def on_send(self, _channel):
    """ Publishes data """
    logger.info("in on_send...")
    self.channel.basic_publish(self.exchange, self.routing_key, 'HELLO from ML Models',
                               pika.BasicProperties(content_type='text/plain',
                                                    delivery_mode=pika.DeliveryMode.Persistent,
                                                    timestamp=int(datetime.now().timestamp())))


def on_receive(self, header, body):
    logger.info(f"Received message...{body.decode('ascii')}")


def process():
    logger.info("Just want to test ML Models...")
    time.sleep(5)
    logger.info("Still works!")
    return True
