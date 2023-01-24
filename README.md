# Sample App for mlmodel
### NOTE: The following is a high level guide; please check back for more complete documentation on how to test this application.

## Pre-requisites:
* Deploy an instance of <a href="https://spring.io/guides/gs/spring-cloud-dataflow/" target="_blank">Spring Cloud Data Flow</a>
* Deploy an instance of <a href="https://docs.ray.io/en/latest/serve/production-guide/kubernetes.html" target="_blank">Ray Cluster</a>
* Deploy an instance of an <a href="https://mlflow.org/docs/latest/quickstart.html" target="_blank">MlFlow Server</a> 
  with an <a href="https://mlflow.org/docs/latest/tracking.html#scenario-4-mlflow-with-remote-tracking-server-backend-and-artifact-stores" target="_blank">S3-compatible backing store</a> and 
  <a href="https://mlflow.org/docs/latest/tracking.html#scenario-5-mlflow-tracking-server-enabled-with-proxied-artifact-storage-access" target="_blank">proxied storage access</a>
* Deploy an external instance of <a href="https://www.rabbitmq.com/kubernetes/operator/operator-overview.html" target="_blank">RabbitMQ cluster</a> (outside of Spring Cloud Data Flow)
* Tested on <a href="https://kubernetes.io/" target="_blank">Kubernetes</a>

## Before running tests, ensure that a test bucket called "test" in the Minio Store is available:
* Create a Read-Write bucket "test"
* Upload data/firehose.csv
* Create a RabbitMQ virtual host "qcirbflj"
* Create a Loadbalancer for the external RabbitMQ service:
  * kubectl expose svc/rabbitmq-headless --name rabbitmq-external --port=5672 --target-port=5672 --type=LoadBalancer
  * (On AWS, you may need to increase the newly created LoadBalancer endpoint's timeout to 3600 seconds)
* Deploy the configmap dependency:
```
kubectl delete configmap test-ml-model || true
kubectl create configmap test-ml-model --from-env-file=.env
kubectl rollout restart deployment/skipper
kubectl rollout restart deployment/scdf-server
```

* Prepare the environment for the pipelines:
```
cd /path/to/scdf/model (from https://github.com/agapebondservant/scdf-ml-model)
scripts/prepare-pipeline-environment.sh
cd -
```

* From the <a href="https://github.com/agapebondservant/scdf-ml-model" target="_blank">mlmodel repo</a>, create Secrets under Settings -> Secrets:
  * Create a Secret named CREATE_PIPELINE_CMD with the content of scripts/commands/create-scdf-ml-pipeline.txt from the mlmodel repo.
  * Create a Secret named UPDATE_PIPELINE_CMD with the content of scripts/commands/update-scdf-ml-pipeline.txt from the mlmodel repo.
