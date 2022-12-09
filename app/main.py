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
from sklearn.datasets import load_iris
from sklearn import tree
import os
import ray
from app.controllers import ScaledTaskController

HttpHealthServer.run_thread()
logger = logging.getLogger('mlmodeltest')
buffer = []
dataset = None
ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ)}) if not ray.is_initialized() else True


@scdf_adapter(environment=None)
def process(msg):
    global buffer, dataset

    client = MlflowClient()
    controller = ScaledTaskController.remote()

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
            baseline_model = ray.get(controller.load_model.remote(model_uri=f'models:/baseline_model/{version}'))
        else:
            try:
                baseline_model = DummyRegressor(strategy="mean").fit(dataset['x'], dataset['target'])
                logger.info(f"Created new baseline model {baseline_model} - registering model...")
                result = controller.log_model.remote(sk_model=baseline_model,
                                                     artifact_path='baseline_model',
                                                     registered_model_name='baseline_model',
                                                     await_registration_for=None)
                logger.info("Get remote results...")
                logger.info(ray.get(result))
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
        logger.info(
            f"Buffer size not yet large enough to process: expected size {utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200}, actual size {len(buffer)} ")
    logger.info("Completed process().")

    # Can use to send data
    # producer.send_data(msg)
    #######################################################
    # END processing
    #######################################################

    return dataset


@scdf_adapter(environment=None)
def evaluate(data):
    client = MlflowClient()
    controller = ScaledTaskController.remote()

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
            baseline_model = ray.get(controller.load_model.remote(model_uri=f'models:/baseline_model/{version}'))

            # Generate candidate model
            candidate_model = DummyRegressor(strategy="mean").fit(data['x'], data['target'])

            # if model evaluation passes, promote candidate model to "staging", else retain baseline model
            logging.info(
                f"Evaluating baseline vs candidate models: baseline_model={baseline_model}, candidate_model={candidate_model}, version={version}")
            result = controller.evaluate_models.remote(baseline_model=baseline_model, candidate_model=candidate_model,
                                                       data=data, version=version)
            logger.info("Get remote results...")
            logger.info(ray.get(result))
            logger.info("Baseline model updated successfully.")

            # Publish ML metrics
            exporter.prepare_counter('candidatemodel:deploynotification',
                                     'New Candidate Model Readiness Notification', mlflow.active_run().data.tags, 1)

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
