import kfp.dsl as dsl
from kfp import components
import kfp.compiler as compiler


@dsl.pipeline(name="launch-kubeflow-pytorchjob",
              description="An example to launch pytorch.")
def ilab_train(
    namespace: str = "mcliffor",
    worker_replicas: int = 1,
    ttl_seconds_after_finished: int = -1,
    job_timeout_minutes: int = 600,
    delete_after_done: bool = False):

    pytorchjob_launcher_op = components.load_component_from_file("component.yaml")

    master = {
        "replicas": 1,
        "restartPolicy": "OnFailure",
        "template": {
            "metadata": {
                "annotations": {
                    # See https://github.com/kubeflow/website/issues/2011
                    "sidecar.istio.io/inject": "false"
                }
            },
            "spec": {
                "containers": [
                    {
                        # To override default command
                       "command": [
                            '/bin/bash',
                            '-c',
                            '--'
                            ],
                        "args": [
                            "python3.11 -u run.py"
                            ],
                        # Or, create your own image from
                        # https://github.com/kubeflow/pytorch-operator/tree/master/examples/mnist
                        "image": "quay.io/michaelclifford/test-train:0.0.11",
                        "name": "pytorch",
                        "resources": {
                            "requests": {
                                "memory": "8Gi",
                                "cpu": "2000m",
                                # Uncomment for GPU
                                "nvidia.com/gpu": 1,
                            },
                            "limits": {
                                "memory": "8Gi",
                                "cpu": "2000m",
                                # Uncomment for GPU
                                "nvidia.com/gpu": 1,
                            },
                        },
                    }
                ],
                # If imagePullSecrets required
                # "imagePullSecrets": [
                #     {"name": "image-pull-secret"},
                # ],
            },
        },
    }

    worker = {}
    if worker_replicas > 0:
        worker = {
            "replicas": worker_replicas,
            "restartPolicy": "OnFailure",
            "template": {
                "metadata": {
                    "annotations": {
                        "sidecar.istio.io/inject": "false"
                    }
                },
                "spec": {
                    "containers": [
                        {   "command": [
                            '/bin/bash',
                            '-c',
                            '--'
                            ],
                            "args": [
                            "python3.11 -u run.py"
                            ],
                            "image": "quay.io/michaelclifford/test-train:0.0.11",
                            "name": "pytorch",
                            "resources": {
                                "requests": {
                                    "memory": "8Gi",
                                    "cpu": "2000m",
                                    # Uncomment for GPU
                                    "nvidia.com/gpu": 1,
                                },
                                "limits": {
                                    "memory": "8Gi",
                                    "cpu": "2000m",
                                    # Uncomment for GPU
                                    "nvidia.com/gpu": 1,
                                },
                            },
                        }
                    ]
                },
            },
        }

    # Launch and monitor the job with the launcher
    pytorchjob_launcher_op(
        name="pytorch-job",
        namespace=namespace,
        master_spec=master,
        worker_spec = worker,
        ttl_seconds_after_finished=ttl_seconds_after_finished,
        job_timeout_minutes=job_timeout_minutes,
        delete_after_done=delete_after_done,
        active_deadline_seconds=100,
        backoff_limit=1)


if __name__ == "__main__":
    pipeline_file = "pipeline.yaml"
    print(f"Compiling pipeline as {pipeline_file}")
    compiler.Compiler().compile(ilab_train,
                                pipeline_file)