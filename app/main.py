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
from evidently.test_suite import TestSuite
from evidently.test_preset import RegressionTestPreset
import json
import nest_asyncio
nest_asyncio.apply()

HttpHealthServer.run_thread()
logger = logging.getLogger('mlmodeltest')


def on_send(self, _channel):
    """ Publishes data """
    logger.info("in on_send...")
    self.channel.basic_publish(self.exchange, self.routing_key, (json.dumps(self.data) or 'HELLO from ML Models'),
                               pika.BasicProperties(content_type='text/plain',
                                                    delivery_mode=pika.DeliveryMode.Persistent,
                                                    timestamp=int(datetime.now().timestamp())))


def on_receive(self, header, body):
    logger.info(f"Received message...{body.decode('ascii')}")


@scdf_adapter(environment=None)
def process(msg):
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

    # Can use to send data
    producer.send_data(msg)  # Can also use to send more data/resend data

    time.sleep(5)

    #######################################################
    # Consumer code:
    #######################################################
    # Start consuming messages
    consumer = ports.get_rabbitmq_port('consumer',
                                       ports.FlowType.INBOUND,
                                       prefetch_count=0,
                                       receive_callback=on_receive)

    # Generate random data
    x = np.linspace(0, np.pi * 8, num=1000)
    y = np.sin(x) + np.random.randint(0, 100)
    dataset = pd.DataFrame({'x': x, 'xlabel': f"Hello, {msg}", 'target': y, 'prediction': y+(np.random.random()*1.5)})

    # Generate Regression report
    old_dataset_path = await utils.get_mlflow_artifacts_inbound_port(artifact_name='old_dataset')
    old_dataset = pd.read_json(old_dataset_path) if old_dataset_path else None
    logger.info(f"downloaded old_dataset...{old_dataset}")
    old_dataset = old_dataset.copy() if old_dataset else dataset.copy()
    dataset['prediction'] = old_dataset['prediction'] + np.random.random()
    logger.info(f"Datasets:\nNew Dataset: {dataset}\nOld Dataset: {old_dataset}")

    # Log Custom ML Metrics
    msg_weight = randrange(0, 101)
    mlflow.log_metric('msg_weight', msg_weight)
    mse = mean_squared_error(dataset['target'], dataset['prediction'])
    mlflow.log_metric('mse', mse)
    logger.info(f"Logging Custom ML metrics - msg_weight...{msg_weight}, mse...{mse}")

    # Perform Evidently Tests
    tests = TestSuite(tests=[
        RegressionTestPreset()
    ])
    tests.run(reference_data=dataset, current_data=old_dataset)
    tests_results_json = tests.json()
    logger.info(f"Evidently generated results...{tests_results_json}")

    # Upload artifacts
    mlflow.log_dict(json.loads(tests_results_json), 'test_results.json')
    mlflow.log_dict(old_dataset.to_dict(), 'old_dataset')
    tests.save_html('/tmp/test_results.html')
    mlflow.log_artifact("/tmp/test_results.html")

    # Publish ML metrics
    logger.info(f"Exporting ML metric - msg_weight...{msg_weight}")
    exporter.prepare_histogram('msg_weight', 'Message Weight', {}, msg_weight)
    exporter.prepare_histogram('mse', 'Mean Squared Error', mlflow.active_run().data.tags, mse)

    logger.info("Completed process().")

    return dataset
