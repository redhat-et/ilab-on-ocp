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
    skills_processed_data: dsl.Output[dsl.Dataset],
    knowledge_processed_data: dsl.Output[dsl.Dataset],
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
    skill_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model.path,
        data_path=f"{sdg.path}/skills_train_msgs*.jsonl",
        data_output_dir=skills_processed_data.path,
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

    knowledge_training_args = TrainingArgs(
        # define data-specific arguments
        model_path=model.path,
        data_path=f"{sdg.path}/knowledge_train_msgs*.jsonl",
        data_output_dir=knowledge_processed_data.path,
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

    data_processing(train_args=skill_training_args)
    data_processing(train_args=knowledge_training_args)


@dsl.component(base_image=PYTHON_IMAGE)
def pytorchjob_manifest_op(
    model_pvc_name: str,
    input_pvc_name: str,
    output_pvc_name: str,
    name_suffix: str,
    # path_to_model: str,
    phase_num: int,
    nproc_per_node: int = 3,
    nnodes: int = 2,
    num_epochs: int = 2,
    effective_batch_size: int = 3840,
    learning_rate: float = 1e-4,
    num_warmup_steps: int = 800,
    save_samples: int = 0,
    max_batch_len: int = 20000,
    seed: int = 42,
) -> NamedTuple("outputs", manifest=str, name=str):
    import inspect
    import os

    def list_phase1_final_model():
        model_dir = "/output/phase_1/model/hf_format"
        models = os.listdir(model_dir)
        newest_idx = max(
            (os.path.getmtime(f"{model_dir}/{model}"), i)
            for i, model in enumerate(models)
        )[-1]
        newest_model = models[newest_idx]
        return f"{model_dir}/{newest_model}"

    Outputs = NamedTuple("outputs", manifest=str, name=str)
    name = f"train-phase-{phase_num}-{name_suffix.rstrip('-sdg')}"

    if phase_num == 1:
        path_to_model = "/input_model/model"
        path_to_data = "/input_data/knowledge_processed_data/data.jsonl"
    elif phase_num == 2:
        path_to_model = list_phase1_final_model()
        path_to_data = "/input_data/skills_processed_data/data.jsonl"

    image = "quay.io/redhat-et/ilab:1.2"

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
                          echo "Running phase {phase_num}"
                          echo "Using {path_to_model} model for training"
                          echo "Using {path_to_data} data for training"
                          mkdir -p /output/phase_{phase_num}/model;
                          mkdir -p /output/data;
                          torchrun --nnodes {nnodes} \
                              --nproc_per_node {nproc_per_node} \
                              --node_rank \$(RANK) \
                              --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) \
                              -m instructlab.training.main_ds \
                              --model_name_or_path={path_to_model} \
                              --data_path={path_to_data} \
                              --output_dir=/output/phase_{phase_num}/model \
                              --num_epochs={num_epochs} \
                              --effective_batch_size={effective_batch_size} \
                              --learning_rate={learning_rate} \
                              --num_warmup_steps={num_warmup_steps} \
                              --save_samples={save_samples} \
                              --log_level=INFO \
                              --max_batch_len={max_batch_len} \
                              --seed={seed} \
                              --cpu_offload_optimizer \
                              --cpu_offload_params \
                              --distributed_training_framework fsdp \
                              --is_granite \
                              --checkpoint_at_epoch
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
                        - name: XDG_CACHE_HOME
                          value: /tmp
                        - name: TRITON_CACHE_DIR
                          value: /tmp
                        - name: HF_HOME
                          value: /tmp
                        - name: TRANSFORMERS_CACHE
                          value: /tmp
                      resources:
                        requests:
                          cpu: 8
                          "nvidia.com/gpu": {nproc_per_node}
                        limits:
                          cpu: 8
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
                          echo "Running phase {phase_num}"
                          echo "Using {path_to_model} model for training"
                          echo "Using {path_to_data} data for training"
                          mkdir -p /tmp/model;
                          torchrun --nnodes {nnodes} \
                            --nproc_per_node {nproc_per_node} \
                            --node_rank \$(RANK) \
                            --rdzv_endpoint \$(MASTER_ADDR):\$(MASTER_PORT) \
                            -m instructlab.training.main_ds \
                            --model_name_or_path={path_to_model} \
                            --data_path={path_to_data} \
                            --output_dir=/tmp/model \
                            --num_epochs={num_epochs} \
                            --effective_batch_size={effective_batch_size} \
                            --learning_rate={learning_rate} \
                            --num_warmup_steps={num_warmup_steps} \
                            --save_samples={save_samples} \
                            --log_level=INFO \
                            --max_batch_len={max_batch_len} \
                            --seed={seed} \
                            --cpu_offload_optimizer \
                            --cpu_offload_params \
                            --distributed_training_framework fsdp \
                            --is_granite \
                            --checkpoint_at_epoch
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
                        - name: XDG_CACHE_HOME
                          value: /tmp
                        - name: TRITON_CACHE_DIR
                          value: /tmp
                        - name: HF_HOME
                          value: /tmp
                        - name: TRANSFORMERS_CACHE
                          value: /tmp
                      resources:
                        requests:
                          cpu: 8
                          "nvidia.com/gpu": {nproc_per_node}
                        limits:
                          cpu: 8
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
