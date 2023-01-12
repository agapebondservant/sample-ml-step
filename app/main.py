from scdfutils import utils, ports
from scdfutils.run_adapter import scdf_adapter
import logging
from scdfutils.http_status_server import HttpHealthServer
from mlmetrics import exporter
import mlflow
from sklearn.dummy import DummyClassifier
import os
import ray
from distributed.ray.distributed import ScaledTaskController
from prodict import Prodict
import json
from datetime import datetime
import app.sentiment_analysis

HttpHealthServer.run_thread()
logger = logging.getLogger('mlmodeltest')
buffer = []
dataset = None
ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ),
                      'excludes': ['*.jar', '.git*/', 'jupyter/']}) if not ray.is_initialized() else True


@scdf_adapter(environment=None)
def process(msg):
    global buffer, dataset
    controller = ScaledTaskController.remote()
    buffer.append(msg.split(','))
    ready = len(buffer) > (utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200)
    run_id = utils.get_env_var('MLFLOW_RUN_ID')
    experiment_id = utils.get_env_var('MLFLOW_EXPERIMENT_ID')
    parent_run_id = utils.get_parent_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])

    # Print MLproject parameter(s)
    logger.info(
        f"MLflow parameters: ready={ready}, msg={msg}, run_id={run_id}, experiment_id={experiment_id}, parent_run_id={parent_run_id}")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d-%H%M"),
                          nested=True):
        #######################################################
        # BEGIN processing
        #######################################################

        # Once the window size is large enough, start processing
        if ready:
            dataset = utils.initialize_timeseries_dataframe(buffer, 'data/schema.csv')
            dataset = app.sentiment_analysis.prepare_data(dataset)

            # Perform Test-Train Split
            df_train, df_test = app.sentiment_analysis.train_test_split(dataset)

            # Perform tf-idf vectorization
            x_train, x_test, y_train, y_test, vectorizer = app.sentiment_analysis.vectorization(df_train, df_test)

            # Generate model
            baseline_model = app.sentiment_analysis.train(x_train, x_test, y_train, y_test)

            # Store metrics
            app.sentiment_analysis.generate_and_save_metrics(x_train, x_test, y_train, y_test, baseline_model)

            # Save model
            app.sentiment_analysis.save_model(baseline_model)

            # Save vectorizer
            app.sentiment_analysis.save_vectorizer(vectorizer)

            # Upload artifacts
            controller.log_artifact.remote(parent_run_id, dataset, 'dataset_snapshot')

            #######################################################
            # RESET globals
            #######################################################
            buffer = []
            dataset = None
        else:
            logger.info(
                f"Buffer size not yet large enough to process: expected size {utils.get_env_var('MONITOR_SLIDING_WINDOW_SIZE') or 200}, actual size {len(buffer)} ")
        logger.info("Completed process().")

        #######################################################
        # END processing
        #######################################################

        return ready


@scdf_adapter(environment=None)
def evaluate(ready):
    controller = ScaledTaskController.remote()
    run_id = utils.get_env_var('MLFLOW_RUN_ID')
    experiment_id = utils.get_env_var('MLFLOW_EXPERIMENT_ID')
    parent_run_id = utils.get_parent_run_id(experiment_names=[utils.get_env_var('CURRENT_EXPERIMENT')])

    # Print MLproject parameter(s)
    logger.info(
        f"MLflow parameters: ready={ready}, run_id={run_id}, experiment_id={experiment_id}, parent_run_id={parent_run_id}")

    with mlflow.start_run(run_id=run_id, experiment_id=experiment_id, run_name=datetime.now().strftime("%Y-%m-%d-%H%M"),
                          nested=True):
        #######################################################
        # BEGIN processing
        #######################################################
        # Once the data is ready, start processing
        if ready:

            # Load existing baseline model (or generate dummy regressor if no model exists)
            version = utils.get_latest_model_version(name='baseline_model', stages=['None', 'Staging'])

            if version:
                # Get candidate model
                candidate_model = ray.get(controller.load_model.remote(
                    parent_run_id,
                    'sklearn',
                    model_uri=f'models:/baseline_model/{version}'))

                # Get baseline model
                if version > 0:
                    baseline_model = ray.get(controller.load_model.remote(
                        parent_run_id,
                        'sklearn',
                        model_uri=f'models:/baseline_model/{version-1}'))
                else:
                    dummy_data = app.sentiment_analysis.generate_dummy_model_data(num_classes=3, size=1000)
                    baseline_model = DummyClassifier(strategy="uniform").fit(dummy_data['x'], dummy_data['target'])

                # Get validation data
                data = ray.get(controller.load_artifact.remote(parent_run_id,
                                                               'dataset_snapshot',
                                                               artifact_uri=f"runs:/{parent_run_id}/dataset_snapshot",
                                                               dst_path="/parent/app/artifacts"))
                data.index = utils.index_as_datetime(data)

                # Fit baseline model
                # dummy_data['prediction'] = baseline_model.predict()

                # if model evaluation passes, promote candidate model to "staging", else retain baseline model
                logging.info(
                    f"Evaluating baseline vs candidate models: baseline_model={baseline_model}, candidate_model={candidate_model}, version={version}")
                result = controller.evaluate_models.remote(baseline_model=baseline_model,
                                                           candidate_model=candidate_model,
                                                           data=data, version=version)
                logger.info("Get remote results...")
                evaluation_result = ray.get(result)
                logger.info(f"Evaluation result: {evaluation_result}")

                # Publish ML metrics
                scdf_tags = Prodict.from_dict(json.loads(utils.get_env_var('SCDF_RUN_TAGS')))
                exporter.prepare_counter('candidatemodel:deploynotification',
                                         'New Candidate Model Readiness Notification', scdf_tags,
                                         int(evaluation_result))

            else:
                logger.error("Baseline model not found...could not perform evaluation")

        else:
            logger.info(f"Data not yet available for processing.")

        logger.info("Completed process().")
        #######################################################
        # END processing
        #######################################################

        return dataset
