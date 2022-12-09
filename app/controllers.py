import ray
import logging
import mlflow
from mlflow import MlflowClient
from mlflow.models import MetricThreshold
import os

#######################################################
# REMOTE code
#######################################################
logger = logging.getLogger('scaledtasks')
ray.init(runtime_env={'working_dir': ".", 'pip': "requirements.txt",
                      'env_vars': dict(os.environ), 'excludes': '*.jar'}) if not ray.is_initialized() else True


@ray.remote
class ScaledTaskController:
    def log_model(self, **kwargs):
        logger.info("In log_model...")
        mlflow.sklearn.log_model(**kwargs)

    def load_model(self, model_uri=None, **kwargs):
        logger.info("In load_model...")
        return mlflow.sklearn.load_model(model_uri)

    def evaluate_models(self, baseline_model=None, candidate_model=None, data=None, version=None):
        logger.info("In evaluate_models...")
        try:
            client = MlflowClient()

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
            result = mlflow.sklearn.log_model(sk_model=candidate_model,
                                              artifact_path='baseline_model',
                                              registered_model_name='baseline_model',
                                              await_registration_for=None)

            return result
        except BaseException as e:
            logger.error(
                "Candidate model training failed to satisfy configured thresholds...could not promote. Retaining baseline model.")
