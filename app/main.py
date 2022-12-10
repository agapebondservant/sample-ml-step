from scdfutils import utils, ports
from scdfutils.run_adapter import scdf_adapter
import logging
from scdfutils.http_status_server import HttpHealthServer
from mlmetrics import exporter
from random import randrange
import mlflow
from mlflow import MlflowClient
from sklearn.dummy import DummyRegressor
import traceback
import os
import ray
from app.controllers import ScaledTaskController
from prodict import Prodict
import json

HttpHealthServer.run_thread()
logger = logging.getLogger('mlmodeltest')
buffer = []
dataset = None
ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ), 'excludes': ['*.jar', '.git*/']}) if not ray.is_initialized() else True


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
    ready_flag = len(buffer) > (utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200)

    # Once the window size is large enough, start processing
    if ready_flag:
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
        controller.log_dict.remote(dataframe=dataset, dict_name='dataset_snapshot')

        # Publish ML metrics
        logger.info(f"Exporting ML metric - msg_weight...{msg_weight}")
        scdf_tags = Prodict.from_dict(json.loads(utils.get_env_var('SCDF_RUN_TAGS')))
        exporter.prepare_histogram('msg_weight', 'Message Weight', scdf_tags, msg_weight)

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

    return ready_flag


@scdf_adapter(environment=None)
def evaluate(ready):
    client = MlflowClient()
    controller = ScaledTaskController.remote()
    run_id = utils.get_env_var('MLFLOW_RUN_ID')

    # Print MLproject parameter(s)
    logger.info(f"Here now...MLflow parameters: ready={ready}, run_id={run_id}")

    #######################################################
    # BEGIN processing
    #######################################################
    # Once the data is ready, start processing
    if ready:

        # Load existing baseline model (or generate dummy regressor if no model exists)
        version = utils.get_latest_model_version(name='baseline_model', stages=['None'])

        if version:
            baseline_model = ray.get(controller.load_model.remote(model_uri=f'models:/baseline_model/{version}'))
            data = ray.get(controller.get_dataframe_from_dict.remote(run_id=run_id, artifact_name='dataset_snapshot'))
            data.index = utils.index_as_datetime(data)

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
            scdf_tags = Prodict.from_dict(json.loads(utils.get_env_var('SCDF_RUN_TAGS')))
            exporter.prepare_counter('candidatemodel:deploynotification',
                                     'New Candidate Model Readiness Notification', scdf_tags, 1)

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
