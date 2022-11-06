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


#######################################################
# Producer code:
#######################################################


def on_send(self, _channel):
    """ Publishes data """
    logging.info("in on_send...")
    self.channel.basic_publish(self.exchange, self.routing_key, 'HELLO from ML Models',
                               pika.BasicProperties(content_type='text/plain',
                                                    delivery_mode=pika.DeliveryMode.Persistent,
                                                    timestamp=int(datetime.now().timestamp())))


producer = ports.get_rabbitmq_port('producer',
                                   ports.FlowType.OUTBOUND,
                                   send_callback=on_send)

# Start publishing messages
# producer.start()

time.sleep(5)

# Can use to send more data
producer.send_data('Hello again!')  # Can use to send more data


#######################################################
# Consumer code:
#######################################################


def on_receive(self, header, body):
    logging.info(f"Received message...{body.decode('ascii')}")


consumer = ports.get_rabbitmq_port('consumer',
                                   ports.FlowType.INBOUND,
                                   prefetch_count=0,
                                   receive_callback=on_receive)

time.sleep(5)

# Start consuming messages
# consumer.start()
