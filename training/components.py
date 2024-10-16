# type: ignore
# pylint: disable=import-outside-toplevel,missing-function-docstring

from typing import NamedTuple, Optional

from kfp import dsl

from utils.consts import PYTHON_IMAGE


@dsl.component(
    base_image=PYTHON_IMAGE,
    packages_to_install=[
        "instructlab-training@git+https://github.com/instructlab/training.git"
    ],
)
def data_processing_op(
    sdg: dsl.Input[dsl.Dataset],
    processed_data: dsl.Output[dsl.Dataset],
    model: dsl.Input[dsl.Artifact],
    max_seq_len: Optional[int] = 4096,
    max_batch_len: Optional[int] = 20000,
):
    import os

    import instructlab.training.data_process as dp
    from instructlab.training import (
        DataProcessArgs,
        TrainingArgs,
    )

    # define training-specific arguments
    training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model.path,
        data_path=f"{sdg.path}/*_train_msgs*.jsonl",
        data_output_dir=processed_data.path,
        # define model-trianing parameters
        max_seq_len=max_seq_len,
        max_batch_len=max_batch_len,
        # XXX(shanand): We don't need the following arguments
        # for data processing. Added them for now to avoid
        # Pydantic validation errors for TrainingArgs
        ckpt_output_dir="data/saved_checkpoints",
        num_epochs=2,
        effective_batch_size=3840,
        save_samples=0,
        learning_rate=2e-6,
        warmup_steps=800,
        is_padding_free=True,
    )

    def data_processing(train_args: TrainingArgs) -> None:
        # early validation logic here
        if train_args.max_batch_len < train_args.max_seq_len:
            raise ValueError(
                f"the 'max_batch_len' cannot be less than 'max_seq_len': {train_args.max_batch_len=} < {train_args.max_seq_len=}"
            )

            # process the training data
        if not os.path.exists(train_args.data_output_dir):
            os.makedirs(train_args.data_output_dir, exist_ok=True)
        dp.main(
            DataProcessArgs(
                # XXX(osilkin): make a decision here, either:
                #   1. the CLI is fully responsible for managing where the data is written
                #   2. we never cache it and simply write it to a tmp file every time.
                #
                # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
                # where the user has a defined place for new temporary data to be written.
                data_output_path=train_args.data_output_dir,
                model_path=train_args.model_path,
                data_path=train_args.data_path,
                max_seq_len=train_args.max_seq_len,
                chat_tmpl_path=train_args.chat_tmpl_path,
            )
        )

    data_processing(train_args=training_args)


@dsl.component(base_image=PYTHON_IMAGE)
def pytorchjob_manifest_op(
    model_pvc_name: str,
    input_pvc_name: str,
    output_pvc_name: str,
    name_suffix: str,
    # path_to_model: str,
    phase_name: str,
    nproc_per_node: int = 3,
    nnodes: int = 2,
) -> NamedTuple("outputs", manifest=str, name=str):
    import inspect
    import os

    def list_phase1_final_model():
        model_dir = "/output/model/hf_format"
        models = os.listdir(model_dir)
        newest_idx = max(
            (os.path.getmtime(f"{model_dir}/{model}"), i)
            for i, model in enumerate(models)
        )[-1]
        newest_model = models[newest_idx]
        return f"{model_dir}/{newest_model}"

    Outputs = NamedTuple("outputs", manifest=str, name=str)
    name = f"train-{phase_name}-{name_suffix.rstrip('-sdg')}"
    if phase_name == "first":
        path_to_model = "/input_model/model"
    elif phase_name == "second":
        path_to_model = list_phase1_final_model()
    image = "registry.redhat.io/rhelai1/instructlab-nvidia-rhel9:1.2"

    manifest = inspect.cleandoc(
        f"""
        apiVersion: kubeflow.org/v1
        kind: PyTorchJob
        metadata:
          name: {name}
        spec:
          nprocPerNode: \\"{nproc_per_node}\\"
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
                          export XDG_CACHE_HOME=/tmp
                          export TRITON_CACHE_DIR=/tmp
                          export HF_HOME=/tmp
                          export TRANSFORMERS_CACHE=/tmp
                          torchrun --nnodes {nnodes} --nproc_per_node {nproc_per_node} --node_rank \$(RANK) --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) -m instructlab.training.main_ds --model_name_or_path={path_to_model} --data_path=/input_data/processed_data/data.jsonl --output_dir=/output/model --num_epochs=2 --effective_batch_size=3840 --learning_rate=1e-4 --num_warmup_steps=800 --save_samples=0 --log_level=INFO --max_batch_len=20000 --seed=42 --cpu_offload_optimizer --distributed_training_framework fsdp --is_granite --checkpoint_at_epoch
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
                          value: \\"{nproc_per_node}\\"
                      resources:
                        requests:
                          cpu: 2
                          "nvidia.com/gpu": {nproc_per_node}
                        limits:
                          cpu: 2
                          "nvidia.com/gpu": {nproc_per_node}
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
                  containers:
                    - args:
                        - |
                          mkdir -p /tmp/model;
                          export TRITON_CACHE_DIR=/tmp
                          export XDG_CACHE_HOME=/tmp
                          export HF_HOME=/tmp
                          export TRANSFORMERS_CACHE=/tmp
                          torchrun --nnodes {nnodes} --nproc_per_node {nproc_per_node} --node_rank \$(RANK) --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) -m instructlab.training.main_ds --model_name_or_path={path_to_model}  --data_path=/input_data/processed_data/data.jsonl --output_dir=/tmp/model --num_epochs=2 --effective_batch_size=3840 --learning_rate=1e-4 --num_warmup_steps=800 --save_samples=0 --log_level=INFO --max_batch_len=20000 --seed=42 --cpu_offload_optimizer --distributed_training_framework fsdp --is_granite --checkpoint_at_epoch
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
                          readOnly: true
                      env:
                        - name: NNODES
                          value: \\"{nnodes}\\"
                        - name: NPROC_PER_NODE
                          value: \\"{nproc_per_node}\\"
                      resources:
                        requests:
                          cpu: 2
                          "nvidia.com/gpu": {nproc_per_node}
                        limits:
                          cpu: 2
                          "nvidia.com/gpu": {nproc_per_node}
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
        """
    )

    return Outputs(manifest, name)
