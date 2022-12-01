from scdfutils import utils, ports
from scdfutils.run_adapter import scdf_adapter
import pika
from datetime import datetime
import logging
import time
from scdfutils.http_status_server import HttpHealthServer
from mlmetrics import exporter
from random import randrange
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import json
from os.path import exists

HttpHealthServer.run_thread()
logger = logging.getLogger('mlmodeltest')
buffer = []
dataset = None

#######################################################
# Producer code: Callback for producer
#######################################################


def on_send(self, _channel):
    """ Publishes data """
    logger.info("in on_send...")
    self.channel.basic_publish(self.exchange, self.routing_key, (json.dumps(self.data) or 'HELLO from ML Models'),
                               pika.BasicProperties(content_type='text/plain',
                                                    delivery_mode=pika.DeliveryMode.Persistent,
                                                    timestamp=int(datetime.now().timestamp())))

#######################################################
# Consumer code: Callback for consumer
#######################################################


def on_receive(self, header, body):
    global buffer
    data = body.decode('ascii')
    logger.info(f"Received message...{data}")
    buffer.append(data)


@scdf_adapter(environment=None)
def process(msg):

    global buffer, dataset

    # Print MLproject parameter(s)
    logger.info(f"MLflow parameters: {msg}")

    #######################################################
    # Producer code:
    #######################################################
    # Start publishing messages
    producer = ports.get_rabbitmq_port('producer',
                                       ports.FlowType.OUTBOUND,
                                       send_callback=on_send)

    time.sleep(5)

    #######################################################
    # Consumer code:
    #######################################################
    # Start consuming messages
    consumer = ports.get_rabbitmq_port('consumer',
                                       ports.FlowType.INBOUND,
                                       prefetch_count=0,
                                       receive_callback=on_receive)

    # Generate dataset
    if len(buffer) > utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE'):
        with open("data/schema.csv", "r") as f:
            columns = f.read().split(',')
            logger.info(f"Columns in schema: {columns}")
        dataset = pd.DataFrame(data=buffer, columns=columns)

        # Log Custom ML Metrics
        msg_weight = randrange(0, 101)
        mlflow.log_metric('msg_weight', msg_weight)
        mse = mean_squared_error(dataset['target'], dataset['prediction'])
        mlflow.log_metric('mse', mse)
        logger.info(f"Logging Custom ML metrics - msg_weight...{msg_weight}, mse...{mse}")

        # Upload artifacts
        mlflow.log_dict(dataset.to_dict(), 'old_dataset')

        # Publish ML metrics
        logger.info(f"Exporting ML metric - msg_weight...{msg_weight}")
        exporter.prepare_histogram('msg_weight', 'Message Weight', {}, msg_weight)
        exporter.prepare_histogram('mse', 'Mean Squared Error', mlflow.active_run().data.tags, mse)

    logger.info("Completed process().")
    buffer = []

    # Can use to send data
    producer.send_data(msg)

    return dataset
