# type: ignore
# pylint: disable=import-outside-toplevel,missing-function-docstring

from typing import Optional

from kfp import dsl

from utils.consts import RHELAI_IMAGE, TOOLBOX_IMAGE


@dsl.component(
    base_image=RHELAI_IMAGE,
    install_kfp_package=False,
)
def data_processing_op(
    model_path: str = "/model",
    sdg_path: str = "/data/sdg",
    skills_path: str = "/data/skills",
    knowledge_path: str = "/data/knowledge",
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
        model_path=model_path,
        data_path=f"{sdg_path}/skills_train_msgs*.jsonl",
        data_output_dir=skills_path,
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
        model_path=model_path,
        data_path=f"{sdg_path}/knowledge_train_msgs*.jsonl",
        data_output_dir=knowledge_path,
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


@dsl.container_component
def skills_processed_data_to_artifact_op(
    skills_processed_data: dsl.Output[dsl.Dataset],
    pvc_path: str = "/data/skills",
):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {skills_processed_data.path}"],
    )


@dsl.container_component
def knowledge_processed_data_to_artifact_op(
    knowledge_processed_data: dsl.Output[dsl.Dataset],
    pvc_path: str = "/data/knowledge",
):
    return dsl.ContainerSpec(
        TOOLBOX_IMAGE,
        ["/bin/sh", "-c"],
        [f"cp -r {pvc_path} {knowledge_processed_data.path}"],
    )


# Change base image to the RHOAI python image with kubeflow_training once available
@dsl.component(base_image="quay.io/redhat-et/ilab:shrey", install_kfp_package=False)
def pytorch_job_launcher_op(
    pytorchjob_output_yaml: dsl.Output[dsl.Artifact],
    model_pvc_name: str,
    input_pvc_name: str,
    output_pvc_name: str,
    name_suffix: str,
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
    job_timeout: int = 86400,
    delete_after_done: bool = False,
):
    import logging
    import os
    import time

    from kubeflow.training import TrainingClient, models
    from kubeflow.training.utils import utils as kfto_utils

    def list_phase1_final_model():
        model_dir = "/output/phase_1/model/hf_format"
        model_list = os.listdir(model_dir)
        newest_idx = max(
            (os.path.getmtime(f"{model_dir}/{model}"), i)
            for i, model in enumerate(model_list)
        )[-1]
        newest_model = model_list[newest_idx]
        return f"{model_dir}/{newest_model}"

    if phase_num == 1:
        path_to_model = "/input_model"
        path_to_data = "/input_data/knowledge/data.jsonl"
    elif phase_num == 2:
        path_to_model = list_phase1_final_model()
        path_to_data = "/input_data/skills/data.jsonl"
    else:
        raise RuntimeError(f"Unsupported value of {phase_num=}")

    resources_per_worker = {"nvidia.com/gpu": nproc_per_node}

    base_image = "registry.stage.redhat.io/rhelai1/instructlab-nvidia-rhel9:1.3.1"
    name = f"train-phase-{phase_num}-{name_suffix.rstrip('-sdg')}"
    command = ["/bin/bash", "-c", "--"]

    master_args = [
        f"""echo "Running phase {phase_num}"
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
                            --cpu_offload_params_fsdp \
                            --distributed_training_framework fsdp \
                            --checkpoint_at_epoch
            """
    ]

    worker_args = [
        f"""echo "Running phase {phase_num}"
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
                            --cpu_offload_params_fsdp \
                            --distributed_training_framework fsdp \
                            --checkpoint_at_epoch
            """
    ]

    # Set volumes
    volumes = [
        models.V1Volume(
            name="input-data",
            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                claim_name=input_pvc_name
            ),
        ),
        models.V1Volume(
            name="model",
            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                claim_name=model_pvc_name
            ),
        ),
        models.V1Volume(
            name="output",
            persistent_volume_claim=models.V1PersistentVolumeClaimVolumeSource(
                claim_name=output_pvc_name
            ),
        ),
    ]

    # Set volume mounts
    volume_mounts_master = [
        models.V1VolumeMount(
            mount_path="/input_data", name="input-data", read_only=True
        ),
        models.V1VolumeMount(mount_path="/input_model", name="model", read_only=True),
        models.V1VolumeMount(mount_path="/output", name="output"),
    ]

    volume_mounts_worker = [
        models.V1VolumeMount(
            mount_path="/input_data", name="input-data", read_only=True
        ),
        models.V1VolumeMount(mount_path="/input_model", name="model", read_only=True),
        models.V1VolumeMount(mount_path="/output", name="output", read_only=True),
    ]

    # Set env variables
    env_vars = [
        models.V1EnvVar(name="NNODES", value=f"{nnodes}"),
        models.V1EnvVar(name="NPROC_PER_NODE", value=f"{nproc_per_node}"),
        models.V1EnvVar(name="XDG_CACHE_HOME", value="/tmp"),
        models.V1EnvVar(name="TRITON_CACHE_DIR", value="/tmp"),
        models.V1EnvVar(name="HF_HOME", value="/tmp"),
        models.V1EnvVar(name="TRANSFORMERS_CACHE", value="/tmp"),
    ]

    # Get master and worker container specs
    master_container_spec = kfto_utils.get_container_spec(
        base_image=base_image,
        name="pytorch",
        resources=resources_per_worker,
        volume_mounts=volume_mounts_master,
    )

    # In the next release of kubeflow-training, the command
    # and the args will be a part of kfto_utils.get_container_spec function
    master_container_spec.command = command
    master_container_spec.args = master_args

    master_container_spec.env = env_vars

    worker_container_spec = kfto_utils.get_container_spec(
        base_image=base_image,
        name="pytorch",
        resources=resources_per_worker,
        volume_mounts=volume_mounts_worker,
    )
    worker_container_spec.command = command
    worker_container_spec.args = worker_args
    worker_container_spec.env = env_vars

    # create master pod spec
    master_pod_template_spec = kfto_utils.get_pod_template_spec(
        containers=[master_container_spec],
        volumes=volumes,
    )

    # create worker pod spec
    worker_pod_template_spec = kfto_utils.get_pod_template_spec(
        containers=[worker_container_spec],
        volumes=volumes,
    )

    logging.getLogger(__name__).setLevel(logging.INFO)
    logging.info("Generating job template.")
    logging.info("Creating TrainingClient.")

    # Initialize training client
    # This also finds the namespace from /var/run/secrets/kubernetes.io/serviceaccount/namespace
    # And it also loads the kube config
    training_client = TrainingClient()
    namespace = training_client.namespace

    # Create pytorch job spec
    job_template = kfto_utils.get_pytorchjob_template(
        name=name,
        namespace=namespace,
        worker_pod_template_spec=worker_pod_template_spec,
        master_pod_template_spec=master_pod_template_spec,
        num_workers=nnodes,
        num_procs_per_worker=nproc_per_node,
    )
    # Save the pytorch job yaml for record keeping and debugging
    with open(pytorchjob_output_yaml.path, "w", encoding="utf-8") as f:
        f.write(job_template.to_str())

    # Run the pytorch job
    logging.info(f"Creating PyTorchJob in namespace: {namespace}")
    training_client.create_job(job_template, namespace=namespace)

    expected_conditions = ["Succeeded", "Failed"]
    logging.info(f"Monitoring job until status is any of {expected_conditions}.")

    def wait_for_job_get_logs(
        name: str,
        namespace: str = None,
        job_kind: str = None,
        expected_conditions: list = ["Succeeded"],
        wait_timeout: int = 600,
        polling_interval: int = 15,
        timeout: int = 1000,
    ) -> str:
        log_lines = set()
        for _ in range(round(wait_timeout / polling_interval)):
            # We should get Job only once per cycle and check the statuses.
            job = training_client.get_job(
                name=name,
                namespace=namespace,
                job_kind=job_kind,
                timeout=timeout,
            )
            # Get Job conditions.
            conditions = training_client.get_job_conditions(
                job=job, timeout=timeout, job_kind=job_kind
            )
            # Return Job when it reaches expected condition.
            for expected_condition in expected_conditions:
                if kfto_utils.has_condition(conditions, expected_condition):
                    return conditions

            # Get logs dictionary
            logs_dict, _ = training_client.get_job_logs(
                name=name, namespace=namespace, job_kind=job_kind
            )

            # Stream new log lines
            for key, value in logs_dict.items():
                if key not in log_lines:
                    logging.info(key)
                    log_lines.add(key)

                for line in value.split("\n"):
                    if line not in log_lines:
                        logging.info(line)
                        log_lines.add(line)

            time.sleep(polling_interval)

    wait_for_job_get_logs(
        name=name,
        namespace=namespace,
        job_kind="PyTorchJob",
        expected_conditions=set(expected_conditions),
        wait_timeout=job_timeout,
        timeout=job_timeout,
    )
    if delete_after_done:
        logging.info("Deleting job after completion.")
        training_client.delete_job(name, namespace)
