from typing import NamedTuple
import kfp.dsl as dsl
from kfp import components

@dsl.component(base_image="python:slim")
def create_worker_spec(worker_num: int = 0) -> NamedTuple(
    "CreatWorkerSpec", [("worker_spec", dict)]):
    """
    Creates pytorch-job worker spec
    """
    from collections import namedtuple
    worker = {}
    if worker_num > 0:
        worker = {
            "replicas": worker_num,
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

    worker_spec_output = namedtuple(
        "MyWorkerOutput", ["worker_spec"]
    )
    return worker_spec_output(worker)

@dsl.pipeline(
    name="launch-kubeflow-pytorchjob",
    description="An example to launch pytorch.",
)
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

    worker_spec_create = create_worker_spec(worker_num=worker_replicas)

    # Launch and monitor the job with the launcher
    pytorchjob_launcher_op(
        name="pytorch-job",
        namespace=namespace,
        master_spec=master,
        worker_spec = worker_spec_create.outputs["worker_spec"],
        ttl_seconds_after_finished=ttl_seconds_after_finished,
        job_timeout_minutes=job_timeout_minutes,
        delete_after_done=delete_after_done,
        active_deadline_seconds=100,
        backoff_limit=1
    )


if __name__ == "__main__":
    import kfp.compiler as compiler

    pipeline_file = "pipeline.yaml"
    print(
        f"Compiling pipeline as {pipeline_file}"
    )
    compiler.Compiler().compile(
        ilab_train, pipeline_file
    )