# type: ignore
# pylint: disable=import-outside-toplevel,missing-function-docstring

from kfp import dsl
from typing import NamedTuple
from utils.consts import PYTHON_IMAGE
from typing import Optional

DATA_IMAGE = 'quay.io/shanand/data_processing:0.0.2'

@dsl.component(base_image=DATA_IMAGE)
def data_processing_op(
    sdg: dsl.Input[dsl.Dataset],
    processed_data: dsl.Output[dsl.Dataset],
    model: dsl.Input[dsl.Artifact],
    max_seq_len: Optional[int] = 4096,
    max_batch_len: Optional[int] = 20000
):
    from run_data_processing import data_processing
    from instructlab.training import TrainingArgs
        # define training-specific arguments
    training_args = TrainingArgs(
        # define data-specific arguments
        model_path = model.path,
        data_path = f"{sdg.path}/*_train_msgs*.jsonl",
        data_output_dir = processed_data.path,

        # define model-trianing parameters
        max_seq_len = max_seq_len,
        max_batch_len = max_batch_len,

       # XXX(shanand): We don't need the following arguments 
       # for data processing. Added them for now to avoid
       # Pydantic validation errors for TrainingArgs
        ckpt_output_dir = "data/saved_checkpoints",
        num_epochs = 2,
        effective_batch_size = 3840,
        save_samples = 0,
        learning_rate = 2e-6,
        warmup_steps = 800,
        is_padding_free = True,
    )

    data_processing(train_args=training_args)

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

    image = 'quay.io/shanand/test-train:0.0.4'
    nprocPerNode = 1
    nnodes = 2

    manifest = inspect.cleandoc(
        f"""
        apiVersion: kubeflow.org/v1
        kind: PyTorchJob
        metadata:
          name: {name}
        spec:
          nprocPerNode: \\"{nprocPerNode}\\"
          pytorchReplicaSpecs:
            Master:
              replicas: 1
              restartPolicy: OnFailure
              template:
                metadata:
                  annotations:
                    sidecar.istio.io/inject: 'false'
                spec:
                  nodeSelector:
                    kubernetes.io/hostname: wrk-6
                  containers:
                    - args:
                        - |
                          mkdir -p /output/model;
                          mkdir -p /output/data;
                          python3.11 -u run_main_ds.py --model_path /input_model/model --ckpt_output_dir /output/model --data_output_dir /input_data/processed_data
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
                      env:
                        - name: NNODES
                          value: \\"{nnodes}\\"
                        - name: NPROC_PER_NODE
                          value: \\"{nprocPerNode}\\"
                      resources:
                        requests:
                          cpu: 2
                          "nvidia.com/gpu": {nprocPerNode}
                        limits:
                          cpu: 2
                          "nvidia.com/gpu": {nprocPerNode}
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
              replicas: {nnodes-1}
              restartPolicy: OnFailure
              template:
                metadata:
                  annotations:
                    sidecar.istio.io/inject: 'false'
                spec:
                  nodeSelector:
                    kubernetes.io/hostname: wrk-6
                  containers:
                    - args:
                        - |
                          mkdir -p /tmp/model;
                          python3.11 -u run_main_ds.py --model_path /input_model/model --ckpt_output_dir /tmp/model --data_output_dir /input_data/processed_data
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
                      env:
                        - name: NNODES
                          value: \\"{nnodes}\\"
                        - name: NPROC_PER_NODE
                          value: \\"{nprocPerNode}\\"
                      resources:
                        requests:
                          cpu: 2
                          "nvidia.com/gpu": {nprocPerNode}
                        limits:
                          cpu: 2
                          "nvidia.com/gpu": {nprocPerNode}
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
