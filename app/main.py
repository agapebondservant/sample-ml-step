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
from mlflow import MlflowClient
import pandas as pd
from mlflow.models import MetricThreshold
import json
from sklearn.dummy import DummyRegressor
import traceback

HttpHealthServer.run_thread()
logger = logging.getLogger('mlmodeltest')
buffer = []
dataset = None

#######################################################
# Producer code: Callback for producer
#######################################################


"""def on_send(self, _channel):
    """""" Publishes data """"""
    logger.info("in on_send...")
    self.channel.basic_publish(self.exchange, self.routing_key, (json.dumps(self.data) or 'HELLO from ML Models'),
                               pika.BasicProperties(content_type='text/plain',
                                                    delivery_mode=pika.DeliveryMode.Persistent,
                                                    timestamp=int(datetime.now().timestamp())))"""

#######################################################
# Consumer code: Callback for consumer
#######################################################


"""def on_receive(self, header, body):
    global buffer
    data = body.decode('ascii')
    logger.info(f"Received message...{data}")
    buffer.append(data)"""

"""
def load_ports():
    #######################################################
    # Producer code:
    #######################################################
    # Start publishing messages
    """"""producer = ports.get_rabbitmq_port('producer',
                                       ports.FlowType.OUTBOUND,
                                       send_callback=on_send)""""""

    time.sleep(5)

    #######################################################
    # Consumer code:
    #######################################################
    # Start consuming messages
    """"""consumer = ports.get_rabbitmq_port('consumer',
                                       ports.FlowType.INBOUND,
                                       prefetch_count=0,
                                       receive_callback=on_receive)""""""
    raw_port = ports.get_rabbitmq_port(':firehose_proxy.raw',
                                       ports.FlowType.INBOUND,
                                       prefetch_count=0,
                                       receive_callback=on_receive)
    # processed_port = ports.get_rabbitmq_port(':firehose_proxy.processed',
    #                                        ports.FlowType.INBOUND,
    #                                         prefetch_count=0,
    #                                         receive_callback=on_receive)

    # return ports for raw data and processed data
    return raw_port
"""


@scdf_adapter(environment=None)
def process(msg):
    with mlflow.start_run(experiment_id=utils.get_env_var("MLFLOW_EXPERIMENT_ID"), run_id=utils.get_env_var('MLFLOW_RUN_ID')) as active_run:

        global buffer, dataset

        client = MlflowClient()

        # Print MLproject parameter(s)
        logger.info(f"Here now...MLflow parameters: {msg}")

        # load_ports()

        #######################################################
        # BEGIN processing
        #######################################################
        buffer.append(msg.split(','))

        # Once the window size is large enough, start processing
        if len(buffer) > (utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200):
            dataset = utils.initialize_timeseries_dataframe(buffer, 'data/schema.csv')

            # Generate and store baseline model if it does not already exist
            version = utils.get_latest_model_version(name='baseline_model', stages=['None'])
            logger.info(f"Version...{version}")
            if version:
                baseline_model = mlflow.sklearn.load_model(f'models:/baseline_model/{version}')
            else:
                try:
                    baseline_model = DummyRegressor(strategy="mean").fit(dataset['x'], dataset['target'])
                    logger.info(f"Created new baseline model {baseline_model} - registering model...")
                    mlflow.sklearn.log_model(sk_model=baseline_model,
                                             artifact_path='baseline_model',
                                             registered_model_name='baseline_model',
                                             await_registration_for=None)
                    logger.info("Logged model to Model Registry.")
                except BaseException as e:
                    logger.info("Could not register model", exc_info=True)
                    traceback.print_exc()
                    raise e

            # Log Custom ML Metrics
            msg_weight = randrange(0, 101)
            mlflow.log_metric('msg_weight', msg_weight)
            logger.info(f"Logging Custom ML metrics - msg_weight...{msg_weight}")

            # Upload artifacts
            dataset.index = dataset.index.astype('str')
            mlflow.log_dict(dataset.to_dict(), 'old_dataset')

            # Publish ML metrics
            logger.info(f"Exporting ML metric - msg_weight...{msg_weight}")
            exporter.prepare_histogram('msg_weight', 'Message Weight', mlflow.active_run().data.tags, msg_weight)

            #######################################################
            # RESET globals
            #######################################################
            buffer = []
            dataset = None
        else:
            logger.info(f"Buffer size not yet large enough to process: expected size {utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200}, actual size {len(buffer)} ")
        logger.info("Completed process().")

        # Can use to send data
        # producer.send_data(msg)
        #######################################################
        # END processing
        #######################################################

        return dataset


@scdf_adapter(environment=None)
def evaluate(data):
    with mlflow.start_run(experiment_id=utils.get_env_var("MLFLOW_EXPERIMENT_ID"), run_id=utils.get_env_var('MLFLOW_RUN_ID')) as active_run:

        client = MlflowClient()

        # Print MLproject parameter(s)
        logger.info(f"Here now...MLflow parameters: {data}")

        #######################################################
        # BEGIN processing
        #######################################################
        # Once the data is ready, start processing
        if data:

            # Load existing baseline model (or generate dummy regressor if no model exists)
            version = utils.get_latest_model_version(name='baseline_model', stages=['None'])

            if version:
                baseline_model = mlflow.sklearn.load_model(f'models:/baseline_model/{version}')

                # Generate candidate model
                candidate_model = DummyRegressor(strategy="mean").fit(data['x'], data['target'])

                # if model evaluation passes, promote candidate model to "staging", else retain baseline model
                logging.info(f"Evaluating baseline vs candidate models: version={version}")
                try:
                    mlflow.evaluate(
                        candidate_model.model_uri,
                        data,
                        targets="target",
                        model_type="regressor",
                        validation_thresholds={
                            "r2_score": MetricThreshold(
                                threshold=0.5,
                                min_absolute_change=0.05,
                                min_relative_change=0.05,
                                higher_is_better=True
                            ),
                        },
                        baseline_model=baseline_model.model_uri,
                    )

                    logger.info("Candidate model passed evaluation; promoting to Staging...")

                    client.transition_model_version_stage(
                        name="baseline_model",
                        version=version,
                        stage="Staging"
                    )

                    logger.info("Candidate model promoted successfully.")

                    logging.info("Updating baseline model...")
                    mlflow.sklearn.log_model(sk_model=candidate_model,
                                             artifact_path='baseline_model',
                                             registered_model_name='baseline_model',
                                             await_registration_for=None)

                    logger.info("Baseline model updated successfully.")

                    # Publish ML metrics
                    exporter.prepare_counter('candidatemodel:deploynotification',
                                             'New Candidate Model Readiness Notification', mlflow.active_run().data.tags, 1)

                except Exception as e:
                    logger.error(
                        "Candidate model training failed to satisfy configured thresholds...could not promote. Retaining baseline model.")

            else:
                logger.error("Baseline model not found...could not perform evaluation")

        else:
            logger.info(f"Data not yet available for processing.")

        logger.info("Completed process().")

        # Can use to send data
        # producer.send_data(msg)
        #######################################################
        # END processing
        #######################################################

        return dataset
