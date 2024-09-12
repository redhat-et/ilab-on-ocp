# type: ignore
# pylint: disable=import-outside-toplevel,missing-function-docstring

from kfp import dsl
from typing import NamedTuple
from utils.consts import PYTHON_IMAGE


@dsl.component(base_image=PYTHON_IMAGE)
def pytorchjob_manifest_op(
    model_pvc_name: str,
    input_pvc_name: str,
    output_pvc_name: str,
    name_suffix: str,
) -> NamedTuple("outputs", manifest=str, name=str):
    import inspect

    Outputs = NamedTuple("outputs", manifest=str, name=str)
    name = f"train-{name_suffix.rstrip('-sdg')}"

    image = 'quay.io/tcoufal/ilab-train:latest'

    manifest = inspect.cleandoc(
        f"""
        apiVersion: kubeflow.org/v1
        kind: PyTorchJob
        metadata:
          name: {name}
        spec:
          pytorchReplicaSpecs:
            Master:
              replicas: 1
              restartPolicy: OnFailure
              template:
                metadata:
                  annotations:
                    sidecar.istio.io/inject: 'false'
                spec:
                  containers:
                    - args:
                        - |
                          mkdir -p /output/model;
                          mkdir -p /output/data;
                          python3.11 -u run.py --nnodes 2 --nproc_per_node 2 --node_rank "$(RANK)" --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) --model_path /input_model --data_path /input_data/*_train_msgs*.jsonl --ckpt_output_dir /output/model --data_output_dir /output/data
                      command:
                        - /bin/bash
                        - '-c'
                        - '--'
                      image: {image}
                      name: pytorch
                      volumeMounts:
                        - mountPath: /input_data
                          name: input-data
                          readOnly: true
                        - mountPath: /input_model
                          name: model
                          readOnly: true
                        - mountPath: /output
                          name: output
                      resources:
                        requests:
                          memory: 8Gi
                          cpu: 2
                          "nvidia.com/gpu": 1
                        limits:
                          memory: 8Gi
                          cpu: 2
                          "nvidia.com/gpu": 1
                  volumes:
                    - name: input-data
                      persistentVolumeClaim:
                        claimName: {input_pvc_name}
                    - name: model
                      persistentVolumeClaim:
                        claimName: {model_pvc_name}
                    - name: output
                      persistentVolumeClaim:
                        claimName: {output_pvc_name}
            Worker:
              replicas: 1
              restartPolicy: OnFailure
              template:
                metadata:
                  annotations:
                    sidecar.istio.io/inject: 'false'
                spec:
                  containers:
                    - args:
                        - |
                          python3.11 -u run.py --nnodes 2 --nproc_per_node 2 --node_rank "$(RANK)" --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) --model_path /input_model --data_path /input_data/*_train_msgs*.jsonl --ckpt_output_dir /tmp/model --data_output_dir /tmp/data
                      command:
                        - /bin/bash
                        - '-c'
                        - '--'
                      image: {image}
                      name: pytorch
                      volumeMounts:
                        - mountPath: /input_data
                          name: input-data
                          readOnly: true
                        - mountPath: /input_model
                          name: model
                          readOnly: true
                      resources:
                        requests:
                          memory: 8Gi
                          cpu: 2
                          "nvidia.com/gpu": 1
                        limits:
                          memory: 8Gi
                          cpu: 2
                          "nvidia.com/gpu": 1
                  volumes:
                    - name: input-data
                      persistentVolumeClaim:
                        claimName: {input_pvc_name}
                    - name: model
                      persistentVolumeClaim:
                        claimName: {model_pvc_name}
        """
    )

    return Outputs(manifest, name)
