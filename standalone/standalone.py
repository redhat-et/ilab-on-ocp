#!/usr/bin/env python3

"""
Standalone Distributed training script

This script provides a standalone version of the pipeline.py script, designed to be used when
Kubeflow pipelines are not available.

Usage:
    This script can be executed directly from the command line. Ensure that the Kubernetes client is
    properly configured before running the script.

Dependencies:
    kubernetes: The Kubernetes Python client library.
    click: A package for creating command-line interfaces.

TODO:
    - Make sure ressources get cleaned up after the job is done. (configmap, secret etc) using a
      finalizer.
    - See if we can use KServe to deploy the model and serve it for SDG Data Generation.
      kubernetes_yaml/mixtral_serve/mixtral_serve.yaml
"""

import base64
import json
import logging
import typing
from os import path
from urllib.parse import urlparse

import click
import kubernetes
import kubernetes.client
import kubernetes.client.rest
import kubernetes.config
import kubernetes.utils
import kubernetes.watch
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

DEFAULT_REPO_URL = "https://github.com/instructlab/taxonomy.git"
K8S_NAME = "kfp-model-server"
TOOLBOX_IMAGE = "registry.access.redhat.com/ubi9/toolbox"
DS_IMAGE = "quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8"
RHELAI_IMAGE = "registry.stage.redhat.io/rhelai1/instructlab-nvidia-rhel9:1.2"
DATA_PVC_NAME = "data"
DATA_PVC_MOUNT_PATH = "/data"
DATA_PVC_MODEL_PATH = path.join(DATA_PVC_MOUNT_PATH, "model")
DATA_VOLUME_NAME = "data"
TAXONOMY_PATH = path.join(DATA_PVC_MOUNT_PATH, "taxonomy")
DATA_PVC_OUTPUT_PATH = path.join(DATA_PVC_MOUNT_PATH, "output")
DATA_PVC_OUTPUT_DATA_PATH = path.join(DATA_PVC_OUTPUT_PATH, "data")
PYTORCH_NNODES = 2
# MMLU_SCORES_PATH = "/output/mmlu-results.txt"
MT_BENCH_OUTPUT_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-results.txt")
MT_BENCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-best.txt")
SDG_OBJECT_STORE_SECRET_NAME = "sdg-object-store-credentials"
KFP_MODEL_SERVER_CM = """
# TODO: remove the following line and replace it with the actual ConfigMap/Secret
kind: ConfigMap
apiVersion: v1
metadata:
  name: kfp-model-server
data:
  endpoint: "https://mistral-7b-instruct-v02-sallyom.apps.ocp-beta-test.nerc.mghpcc.org/v1"
  model: "mistral-7b-instruct-v02"
---
apiVersion: v1
kind: Secret
metadata:
  name: kfp-model-server
type: Opaque
stringData:
  api_key: ""

"""

JUDGE_SERVING_NAME = "judge-serving-details"

PYTORCH_TRAINING_JOB = """
apiVersion: kubeflow.org/v1
kind: PyTorchJob
metadata:
  name: {name}
spec:
  nprocPerNode: \"{nproc_per_node}\"
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
                  phase_num={phase_num}
                  echo "Running phase $phase_num"
                  PATH_TO_MODEL={path_to_model}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/hf_format/$(ls --sort=time {path_to_model}/output/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  mkdir -p /data/model;
                  mkdir -p /data/data;
                  mkdir -p {path_to_model}/output
                  export XDG_CACHE_HOME=/tmp
                  export TRITON_CACHE_DIR=/tmp
                  export HF_HOME=/tmp
                  export TRANSFORMERS_CACHE=/tmp
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path=/data/processed_data/data.jsonl \
                    --output_dir={path_to_model}/output \
                    --num_epochs={epoch_num} \
                    --effective_batch_size=3840 \
                    --learning_rate=1e-4 \
                    --num_warmup_steps=800 \
                    --save_samples=0 \
                    --log_level=INFO \
                    --max_batch_len=20000 \
                    --seed=42 \
                    --cpu_offload_optimizer \
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
                - mountPath: /data
                  name: data
              env:
                - name: NNODES
                  value: \"{nnodes}\"
                - name: NPROC_PER_NODE
                  value: \"{nproc_per_node}\"
              resources:
                requests:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
                limits:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: {data_pvc_name}
    Worker:
      replicas: {worker_replicas}
      restartPolicy: OnFailure
      template:
        metadata:
          annotations:
            sidecar.istio.io/inject: 'false'
        spec:
          containers:
            - args:
                - |
                  phase_num={phase_num}
                  echo "Running phase $phase_num"
                  PATH_TO_MODEL={path_to_model}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/hf_format/$(ls --sort=time {path_to_model}/output/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  mkdir -p /tmp/model;
                  export TRITON_CACHE_DIR=/tmp
                  export XDG_CACHE_HOME=/tmp
                  export HF_HOME=/tmp
                  export TRANSFORMERS_CACHE=/tmp
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path=/data/processed_data/data.jsonl \
                    --output_dir=/tmp/model \
                    --num_epochs={epoch_num} \
                    --effective_batch_size=3840 \
                    --learning_rate=1e-4 \
                    --num_warmup_steps=800 \
                    --save_samples=0 \
                    --log_level=INFO \
                    --max_batch_len=20000 \
                    --seed=42 \
                    --cpu_offload_optimizer \
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
                - mountPath: /data
                  name: data
              env:
                - name: NNODES
                  value: \"{nnodes}\"
                - name: NPROC_PER_NODE
                  value: \"{nproc_per_node}\"
              resources:
                requests:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
                limits:
                  cpu: 2
                  "nvidia.com/gpu": {nproc_per_node}
          volumes:
            - name: data
              persistentVolumeClaim:
                claimName: {data_pvc_name}
"""
# TODO: support signature version?
DATA_SCRIPT = """
set -e

FORCE_PULL={force_pull}
if [ -s {data_pvc_mount_path}/data.tar.gz ] && [ -d {data_pvc_mount_path}/data ] && [ -d {data_pvc_mount_path}/model ] ; then
  echo "Data tarball and sdg/model directories already exist in the PVC. Skipping download."
  if [ "$FORCE_PULL" == "None" ] || [ "$FORCE_PULL" == "False" ]; then
    echo "'--force-pull' is not set - will not force pull the data from the object store"
    ls -laR {data_pvc_mount_path}
    exit 0
  else
    echo "'--force-pull' is set to true - will force pull the data from the object store"
  fi
fi

export STRATEGY={strategy}

if [ -z "$STRATEGY" ] || [ "$STRATEGY" == "None" ]; then
  echo "STRATEGY is not set - must be 'download' or 'upload'"
  exit 1
fi

if python3 -c 'import boto3'; then
  echo 'boto3 is already installed'
else
  if ! [ -x "$(command -v pip)" ]; then
    python3 -m ensurepip || python3 -m ensurepip --user || dnf install python3-pip -y
  fi
  python3 -m pip install boto3
fi


tmp=$(mktemp -d)
cat <<EOF > "$tmp"/download_s3.py
import os
import boto3

def str_to_bool(s):
    if s is None:
      return False
    return s.lower() in ['true', '1', 't', 'y', 'yes']

def build_boto3_client():
  return boto3.client(
    's3',
    aws_access_key_id=os.getenv('SDG_OBJECT_STORE_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SDG_OBJECT_STORE_SECRET_KEY'),
    endpoint_url=os.getenv('SDG_OBJECT_STORE_ENDPOINT', None),
    region_name=os.getenv('SDG_OBJECT_STORE_REGION', None),
    verify=str_to_bool(os.getenv('SDG_OBJECT_STORE_VERIFY_TLS', None))
)

def download_s3_file():
    s3 = build_boto3_client()

    bucket_name = os.getenv('SDG_OBJECT_STORE_BUCKET')
    s3_key = os.getenv('SDG_OBJECT_STORE_DATA_KEY')
    output_file = '{data_pvc_mount_path}/data.tar.gz'

    s3.download_file(bucket_name, s3_key, output_file)

def upload_s3_file():
    s3 = build_boto3_client()

    bucket_name = os.getenv('SDG_OBJECT_STORE_BUCKET')
    s3_key = os.getenv('SDG_OBJECT_STORE_DATA_KEY') # TODO: change the name for the model name
    input_file = '{data_pvc_mount_path}/data.tar.gz' # TODO: change for model path

    s3.upload_file(input_file, bucket_name, s3_key)

if __name__ == "__main__":
    if os.getenv('STRATEGY') == 'download':
      print('Downloading file from S3')
      download_s3_file()
    elif os.getenv('STRATEGY') == 'upload':
      print('Uploading file to S3')
      upload_s3_file()
    else:
      raise ValueError('Unknown STRATEGY')
EOF

python "$tmp"/download_s3.py

if [ "$STRATEGY" == "download" ]; then
  # List top-level directories only (no nested directories)
  top_level_dirs=$(tar --exclude='*/*' --list --file {data_pvc_mount_path}/data.tar.gz)

  # Loop through the expected directories and check if they exist in the archive
  for dir in data model taxonomy; do
    if ! echo "$top_level_dirs" | grep -q "^$dir/$"; then
      echo "Archive does not contain a '$dir' directory"
      exit 1
    fi
  done
  echo "All expected directories are present."

  echo "Extracting data from the archive"
  tar -C {data_pvc_mount_path} -xvf {data_pvc_mount_path}/data.tar.gz
  ls -laR {data_pvc_mount_path}
fi
"""

JOB_SCRIPT_EXAMPLE = """
kind: Job
apiVersion: batch/v1
metadata:
  name: {name}
  namespace: {namespace}
spec:
  template:
    spec:
      serviceAccountName: {service_account}
      containers:
      - name: {name}
        image: {image}
        command:
        - "python3"
        - "/config/{script_name}"
        - "run"
        - "--namespace"
        - "{namespace_workflow}"
        - "--storage-class"
        - "{storage_class}"
        - "--sdg-object-store-secret"
        - "{sdg_object_store_secret}"
        volumeMounts:
        - name: script-config
          mountPath: /config
      restartPolicy: Never
      volumes:
      - name: script-config
        configMap:
          name: {script_configmap}
"""

PYTHON_EXECUTOR = """
set -e
export XDG_CACHE_HOME=/tmp
export OUTLINES_CACHE_DIR=/tmp
export NUMBA_CACHE_DIR=/tmp
export TRANSFORMERS_CACHE=/tmp
export HF_HOME=/tmp
export HOME=/tmp
export TRITON_CACHE_DIR=/tmp

tmp=$(mktemp -d)
cat <<EOF > "$tmp"/exec.py

{python_code}

if __name__ == "__main__":
    {python_main}

EOF

python3 "$tmp"/exec.py
"""


@click.group()
def cli():
    """
    Command Line Interface (CLI) entry point.

    This function serves as the main entry point for the command line interface.
    It currently does not perform any operations.
    """


@cli.group(invoke_without_command=True)
@click.option(
    "--namespace",
    type=str,
    default="default",
    help="Kubernetes namespace to run the job",
)
@click.option(
    "--namespace-workflow",
    type=str,
    default="default",
    help="Kubernetes namespace to run the end-to-end workflow that the script will execute",
)
@click.option(
    "--name",
    type=str,
    default="distributed-ilab",
    help="Name of the Job to that can run the script",
)
@click.option(
    "--image",
    type=str,
    help="Image to use to run the script in a Job",
    required=True,
)
@click.option(
    "--service-account",
    type=str,
    default="default",
    help="Service account to use for the Job",
)
@click.option(
    "--script-configmap",
    type=str,
    help="Name of the ConfigMap containing the standalone.py script",
    required=True,
)
@click.option(
    "--script-name",
    type=str,
    help="Name of the standalone script in the ConfigMap",
    default="standalone.py",
)
@click.option(
    "--storage-class",
    type=str,
    help="Storage class to use for the PersistentVolumeClaim - for SDG only",
)
@click.option(
    "--sdg-object-store-secret",
    envvar="SDG_OBJECT_STORE_SECRET",
    help=(
        "Name of the Kubernetes Secret containing the SDG object store credentials. "
        "The namespace is inferred from the namespace option. "
        "The following keys are expected: bucket, access_key, secret_key, data_key. "
        " (SDG_OBJECT_STORE_SECRET env var)"
        "If used, the  endpoint, bucket, access_key, secret_key, region, data_key, verify_tls options will be ignored."
        "All supported options are: endpoint, bucket, access_key, secret_key, region, data_key, verify_tls"
    ),
    default=SDG_OBJECT_STORE_SECRET_NAME,
    type=str,
)
def show(
    namespace: str,
    namespace_workflow: str,
    name: str,
    image: str,
    script_configmap: str,
    script_name: str,
    service_account: str,
    storage_class: str,
    sdg_object_store_secret: str,
):
    """
    Print an example Job YAML to stdout to run the script in a Kubernetes cluster.
    The job excepts the standalone.py script to be available in a ConfigMap.
    """
    print(
        JOB_SCRIPT_EXAMPLE.format(
            name=name,
            namespace=namespace,
            namespace_workflow=namespace_workflow,
            image=image,
            script_configmap=script_configmap,
            script_name=script_name,
            service_account=service_account,
            storage_class=storage_class,
            sdg_object_store_secret=sdg_object_store_secret,
        )
    )


@cli.group(invoke_without_command=True)
@click.option("--namespace", type=str, help="Kubernetes namespace to use")
@click.option(
    "--taxonomy-repo-url",
    type=str,
    default=DEFAULT_REPO_URL,
    help="URL of the taxonomy repository - for SDG only",
    hidden=True,
)
@click.option(
    "--taxonomy-repo-branch",
    type=str,
    help="Branch of the taxonomy repository - for SDG only",
    hidden=True,
)
@click.option(
    "--taxonomy-repo-pr",
    type=str,
    help="Pull request number of the taxonomy repository - for SDG only",
    hidden=True,
)
@click.option(
    "--storage-class",
    type=str,
    help="Storage class to use for the PersistentVolumeClaim - for SDG only",
)
@click.option(
    "--serving-endpoint",
    type=str,
    help="Serving endpoint for SDG - for SDG only",
    hidden=True,
)
@click.option(
    "--serving-model",
    type=str,
    help="Serving model for SDG - for SDG only",
    hidden=True,
)
@click.option(
    "--judge-serving-endpoint",
    type=str,
    help=(
        "Serving endpoint for evaluation."
        "e.g. http://serving.kubeflow.svc.cluster.local:8080/v1"
    ),
    required=True,
)
@click.option(
    "--judge-serving-model-name",
    type=str,
    help="The name of the model to use for evaluation.",
    required=True,
)
@click.option(
    "--judge-serving-model-api-key",
    type=str,
    help=(
        "Serving model API key for evaluation. " "(JUDGE_SERVING_MODEL_API_KEY env var)"
    ),
    envvar="JUDGE_SERVING_MODEL_API_KEY",
    required=True,
)
@click.option(
    "--nproc-per-node",
    type=int,
    help="Number of processes per node - for training only",
    default=1,
)
@click.option(
    "--eval-type",
    help="Type of evaluation to run",
    type=click.Choice(["mt-bench", "mt-bench-branch"]),
    hidden=True,
)
@click.option(
    "--training-phase",
    help="Type of training phase to run",
    type=click.Choice(["1", "2"]),
)
@click.option(
    "--model-to-train",
    help=(
        "Path to model to train (PVC filesystem path). "
        "Useful when calling training phases independently and users wants to point to the epoch directory. "
        "Very advanced usage, not recommended for general use."
    ),
    type=str,
)
@click.option(
    "--sdg-object-store-endpoint",
    envvar="SDG_OBJECT_STORE_ENDPOINT",
    help=(
        "Object store endpoint for SDG if different than the official AWS S3 endpoint. "
        "Expects an URL. TLS with self-signed certificates is not supported. (SDG_OBJECT_STORE_ENDPOINT env var)"
        "e.g. https://s3.openshift-storage.svc:443"
        "Don't forget the URL scheme (http/https) and the port"
    ),
    type=str,
)
@click.option(
    "--sdg-object-store-bucket",
    envvar="SDG_OBJECT_STORE_BUCKET",
    help="Object store bucket containing SDG data. (SDG_OBJECT_STORE_BUCKET env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-access-key",
    envvar="SDG_OBJECT_STORE_ACCESS_KEY",
    help="Object store access key for SDG. (SDG_OBJECT_STORE_ACCESS_KEY env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-secret-key",
    envvar="SDG_OBJECT_STORE_SECRET_KEY",
    help="Object store secret key for SDG. (SDG_OBJECT_STORE_SECRET_KEY env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-region",
    envvar="SDG_OBJECT_STORE_REGION",
    help="Region for the object store. (SDG_OBJECT_STORE_REGION env var)",
    type=str,
)
@click.option(
    "--sdg-object-store-data-key",
    envvar="SDG_OBJECT_STORE_DATA_KEY",
    help=(
        "Name of tarball that contains SDG data AND model files. (SDG_OBJECT_STORE_DATA_KEY env var)."
        "The tarball MUST contain two directories: data and model."
        "The data directory contains the SDG data."
        "The model directory contains the model to train."
        "To archive , use the following command: tar -czvf data.tar.gz /path/to/data /path/to/model ."
    ),
    type=str,
)
@click.option(
    "--sdg-object-store-verify-tls",
    envvar="SDG_OBJECT_STORE_VERIFY_TLS",
    help="Verify TLS for the object store. (SDG_OBJECT_STORE_VERIFY_TLS env var).",
    default=True,
    type=bool,
)
@click.option(
    "--sdg-object-store-secret",
    envvar="SDG_OBJECT_STORE_SECRET",
    help=(
        "Name of the Kubernetes Secret containing the SDG object store credentials. "
        "The namespace is inferred from the namespace option. "
        "The following keys are expected: bucket, access_key, secret_key, data_key. "
        " (SDG_OBJECT_STORE_SECRET env var)"
        "If used, the  endpoint, bucket, access_key, secret_key, region, data_key, verify_tls options will be ignored."
        "All supported options are: endpoint, bucket, access_key, secret_key, region, data_key, verify_tls"
    ),
    type=str,
)
@click.option(
    "--force-pull",
    help="Force pull the data (sdg data and model) from the object store even if it already exists in the PVC.",
    is_flag=True,
    default=False,
)
@click.option(
    "--training-1-epoch-num", help="Number of epochs to train the model for.", default=7
)
@click.option(
    "--training-2-epoch-num",
    help="Number of epochs to train the model for.",
    default=10,
)
@click.pass_context
def run(
    ctx: click.Context,
    namespace: typing.Optional[str] = None,
    taxonomy_repo_url: str = "",
    taxonomy_repo_branch: typing.Optional[str] = "",
    taxonomy_repo_pr: typing.Optional[str] = "",
    storage_class: typing.Optional[str] = None,
    serving_endpoint: typing.Optional[str] = None,
    serving_model: typing.Optional[str] = None,
    judge_serving_endpoint: typing.Optional[str] = None,
    judge_serving_model_name: typing.Optional[str] = None,
    judge_serving_model_api_key: typing.Optional[str] = None,
    nproc_per_node: typing.Optional[int] = 1,
    eval_type: typing.Optional[str] = None,
    training_phase: typing.Optional[str] = None,
    model_to_train: typing.Optional[str] = None,
    sdg_object_store_endpoint: typing.Optional[str] = None,
    sdg_object_store_bucket: typing.Optional[str] = None,
    sdg_object_store_access_key: typing.Optional[str] = None,
    sdg_object_store_secret_key: typing.Optional[str] = None,
    sdg_object_store_region: typing.Optional[str] = None,
    sdg_object_store_data_key: typing.Optional[str] = None,
    sdg_object_store_verify_tls: typing.Optional[bool] = None,
    sdg_object_store_secret: typing.Optional[str] = None,
    force_pull: typing.Optional[bool] = False,
    training_1_epoch_num: int = 7,
    training_2_epoch_num: int = 10,
):
    """
    Execute the distributed training on Kubernetes.

    Args:
        namespace (str): The namespace to use for the setup process.
        taxonomy_repo_url (str): The URL of the taxonomy repository. For SDG only.
        taxonomy_repo_branch (str): The branch of the taxonomy repository. For SDG only.
        taxonomy_repo_pr (int): The pull request number of the taxonomy repository. For SDG only.
        storage_class (str): The storage class to use for the PersistentVolumeClaim. For SDG only.
        serving_endpoint (str): The serving endpoint for SDG. For SDG only.
        serving_model (str): The serving model for SDG. For SDG only.
        judge_serving_endpoint (str): The serving endpoint for evaluation. For Evaluation only.
        judge_serving_model_name (str): The serving model name for evaluation. For Evaluation only.
        judge_serving_model_api_key (str): The serving model API key for evaluation. For Evaluation only.
        nproc_per_node (int): The number of processes per node. For training only.
        eval_type (str): The type of evaluation to run.
        training_phase (str): The type of training phase to run.
        model_to_train (str): The path to model to train (PVC filesystem path).
        sdg_object_store_endpoint (str): The object store endpoint for SDG.
        sdg_object_store_bucket (str): The object store bucket containing SDG data.
        sdg_object_store_access_key (str): The object store access key for SDG.
        sdg_object_store_secret_key (str): The object store secret key for SDG.
        sdg_object_store_region (str): The region for the object store.
        sdg_object_store_data_key (str): The name of the tarball that contains SDG data.
        sdg_object_store_verify_tls (bool): Verify TLS for the object store.
        sdg_object_store_secret (str): The name of the Kubernetes Secret containing the SDG object store credentials. The namespace is inferred from the namespace option.
        force_pull (bool): Force pull the data (sdg data and model) from the object store even if it already exists in the PVC.
        training_1_epoch_num (int): Number of epochs to train the model for during phase 1.
        training_2_epoch_num (int): Number of epochs to train the model for during phase 2.

    Returns:
        None
    """
    ctx.ensure_object(dict)
    ctx.obj["namespace"] = namespace
    ctx.obj["taxonomy_repo_url"] = taxonomy_repo_url
    ctx.obj["taxonomy_repo_branch"] = taxonomy_repo_branch
    ctx.obj["taxonomy_repo_pr"] = taxonomy_repo_pr
    ctx.obj["storage_class"] = storage_class
    ctx.obj["serving_endpoint"] = serving_endpoint
    ctx.obj["serving_model"] = serving_model
    ctx.obj["judge_serving_endpoint"] = judge_serving_endpoint
    ctx.obj["judge_serving_model_name"] = judge_serving_model_name
    ctx.obj["judge_serving_model_api_key"] = judge_serving_model_api_key
    ctx.obj["nproc_per_node"] = nproc_per_node
    ctx.obj["eval_type"] = eval_type
    ctx.obj["training_phase"] = training_phase
    ctx.obj["model_to_train"] = model_to_train
    ctx.obj["sdg_object_store_endpoint"] = sdg_object_store_endpoint
    ctx.obj["sdg_object_store_bucket"] = sdg_object_store_bucket
    ctx.obj["sdg_object_store_access_key"] = sdg_object_store_access_key
    ctx.obj["sdg_object_store_secret_key"] = sdg_object_store_secret_key
    ctx.obj["sdg_object_store_region"] = sdg_object_store_region
    ctx.obj["sdg_object_store_data_key"] = sdg_object_store_data_key
    ctx.obj["sdg_object_store_verify_tls"] = sdg_object_store_verify_tls
    ctx.obj["sdg_object_store_secret"] = sdg_object_store_secret
    ctx.obj["force_pull"] = force_pull
    ctx.obj["training_1_epoch_num"] = training_1_epoch_num
    ctx.obj["training_2_epoch_num"] = training_2_epoch_num

    ##########################
    # MAIN WORKFLOW SEQUENCE #
    ##########################
    # When the script is simply called like: 'python standalone.py run'
    # We will run the entire workflow
    if ctx.invoked_subcommand is None:
        # SDG Full
        # ctx.invoke(sdg)

        # SDG Data Fetch
        ctx.invoke(sdg_data_fetch)

        # Begin multi-phased distributed training
        logger.info("Running multi-phased distributed training.")

        # Training Phase 1
        ctx.obj["training_phase"] = "1"
        ctx.invoke(train)

        # Evaluation of phase 1 with MMLU
        # ctx.obj["eval_type"] = "mmlu"
        # scores = ctx.invoke(evaluation)
        # scores = json.loads(scores)
        # best_model = max(scores, key=lambda x: x["average_score"])
        # logger.info("Best model: %s", best_model.get("model"))
        # ctx.obj["model_to_train"] = best_model.get("model")

        # Training Phase 2
        ctx.obj["training_phase"] = "2"
        ctx.invoke(train)

        # Evaluation of phase 2 with MT-Bench
        ctx.obj["eval_type"] = "mt-bench"
        scores = ctx.invoke(evaluation)
        scores = json.loads(scores)
        logger.info("Best model: %s", scores.get("best_model"))
        ctx.obj["candidate_model"] = scores.get("best_model")

        # Final evaluation
        ctx.obj["eval_type"] = "mt-bench-branch"
        scores = ctx.invoke(evaluation)
        scores = json.loads(scores)
        logger.info("Best model: %s", scores.get("best_model"))
        ctx.obj["candidate_model"] = scores.get("best_model")

        # Push the best model to S3
        # TODO


def get_security_context() -> kubernetes.client.V1SecurityContext:
    """
    Get the security context.
    """
    return kubernetes.client.V1SecurityContext(
        capabilities=kubernetes.client.V1Capabilities(drop=["ALL"]),
        run_as_non_root=True,
    )


def get_vol_mount() -> list[kubernetes.client.V1VolumeMount]:
    """
    Get the volume mount for the SDG job.
    """
    return [
        kubernetes.client.V1VolumeMount(
            name=DATA_VOLUME_NAME, mount_path=DATA_PVC_MOUNT_PATH
        ),
    ]


def get_vol() -> list[kubernetes.client.V1Volume]:
    """
    Get the volume for the SDG job.
    """
    return [
        kubernetes.client.V1Volume(
            name=DATA_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=DATA_PVC_NAME
            ),
        ),
    ]


def create_sdg_job(
    namespace: str,
    job_name: str,
    exec_git_clone_op_repo_url: str = "",
    exec_git_clone_op_repo_branch: str = "",
    exec_git_clone_op_repo_pr: str = "",
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to run SDG steps.

    Steps:
        1. InitContainer to fetch the taxonomy data. - EmptyDir volume to share data between
           containers.
        2. InitContainer to generate synthetic data. - Stored on EmptyDir volume. (Option to push to
           S3?)
        3. Main container to pre-process the data before training. From the EmptyDir volume and copy
           the result to the PVC.
    Args:
        namespace (str): The namespace in which the job will be created.
        job_name (str): The name of the job.
        exec_git_clone_op_repo_url (str): The URL of the taxonomy repository.
        exec_git_clone_op_repo_branch (str, optional): The branch of the taxonomy repository.
        exec_git_clone_op_repo_pr (str, optional): The pull request number of the taxonomy repository.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """
    # Configureate Pod template container
    exec_sdg_op_command = """
from typing import *

def sdg_op(
    num_instructions_to_generate: int,
    taxonomy: str,
    sdg: str,
    repo_branch: Optional[str],
    repo_pr: Optional[int],
):
    from os import getenv

    import openai
    from instructlab.sdg import generate_data
    from instructlab.sdg.utils.taxonomy import read_taxonomy

    api_key = getenv("api_key")
    model = getenv("model")
    endpoint = getenv("endpoint")
    client = openai.OpenAI(base_url=endpoint, api_key=api_key)

    taxonomy_base = "main" if repo_branch or (repo_pr and int(repo_pr) > 0) else "empty"

    print("Generating syntetic dataset for:")
    print()
    print(read_taxonomy(taxonomy, taxonomy_base))

    # generate_data has a magic word for its taxonomy_base argument - 'empty'
    # it allows generating from the whole repo, see:
    # https://github.com/instructlab/sdg/blob/c6a9e74a1618b1077cd38e713b8aaed8b7c0c8ce/src/instructlab/sdg/utils/taxonomy.py#L230
    generate_data(
        client=client,
        num_instructions_to_generate=num_instructions_to_generate,
        output_dir=sdg,
        taxonomy=taxonomy,
        taxonomy_base=taxonomy_base,
        model_name=model,
        chunk_word_count=1000,
        server_ctx_size=4096,
    )
"""
    exec_sdg_op_args = """
sdg_op(num_instructions_to_generate=2, repo_branch="", repo_pr="", taxonomy="/data/taxonomy", sdg="/data/generated")
"""

    exec_huggingface_importer_op_command = """
from typing import *

def huggingface_importer_op(model: str, repo_name: str):
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_name, cache_dir="/tmp", local_dir=model)
"""
    exec_huggingface_importer_op_args = """
huggingface_importer_op(repo_name="ibm-granite/granite-7b-base", model="/data/model")
"""

    exec_data_processing_op_command = """
from typing import *

def data_processing_op(
    sdg: str,
    processed_data: str,
    model: str,
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
        model_path=model,
        data_path=f"{sdg}/*_train_msgs*.jsonl",
        data_output_dir=processed_data,
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
"""
    exec_data_processing_op_args = """
data_processing_op(max_seq_len=4096, max_batch_len=20000, sdg="/data/data", model="/data/model", processed_data="/data/processed_data")
"""

    init_containers = [
        kubernetes.client.V1Container(
            name="sdg-op-fetch-taxonomy-data",
            image="registry.access.redhat.com/ubi9/toolbox",
            command=["/bin/sh", "-c"],
            args=[
                'git clone {exec_git_clone_op_repo_url} {TAXONOMY_PATH} && cd {TAXONOMY_PATH} && if [ -n "{exec_git_clone_op_repo_branch}" ]; then git fetch origin {exec_git_clone_op_repo_branch} && git checkout {exec_git_clone_op_repo_branch}; elif [ -n "{exec_git_clone_op_repo_pr}" ] && [ {exec_git_clone_op_repo_pr} -gt 0 ]; then git fetch origin pull/{exec_git_clone_op_repo_pr}/head:{exec_git_clone_op_repo_pr} && git checkout {exec_git_clone_op_repo_pr}; fi '
            ],
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
        ),
        kubernetes.client.V1Container(
            name="sdg-op-generate-synthetic-data",
            # image="quay.io/tcoufal/ilab-sdg:latest",
            image="registry.redhat.io/rhelai1/instructlab-nvidia-rhel9:1.1-1724960989",
            command=["/bin/sh", "-ce"],
            args=[
                PYTHON_EXECUTOR.format(
                    python_code=exec_sdg_op_command,
                    python_main=exec_sdg_op_args.strip(),
                ),
            ],
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
            env_from=[
                kubernetes.client.V1EnvFromSource(
                    config_map_ref=kubernetes.client.V1ConfigMapEnvSource(name=K8S_NAME)
                ),
                kubernetes.client.V1EnvFromSource(
                    secret_ref=kubernetes.client.V1SecretEnvSource(name=K8S_NAME)
                ),
            ],
        ),
        kubernetes.client.V1Container(
            name="huggingface-importer-op",
            image="registry.access.redhat.com/ubi9/python-311:latest",
            command=["/bin/sh", "-ce"],
            args=[
                PYTHON_EXECUTOR.format(
                    python_code=exec_huggingface_importer_op_command,
                    python_main=exec_huggingface_importer_op_args.strip(),
                ),
            ],
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
            env_from=[
                kubernetes.client.V1EnvFromSource(
                    config_map_ref=kubernetes.client.V1ConfigMapEnvSource(name=K8S_NAME)
                ),
                kubernetes.client.V1EnvFromSource(
                    secret_ref=kubernetes.client.V1SecretEnvSource(name=K8S_NAME)
                ),
            ],
        ),
        kubernetes.client.V1Container(
            name="sdg-preprocess",
            image="registry.access.redhat.com/ubi9/python-311:latest",
            command=["/bin/sh", "-ce"],
            args=[
                PYTHON_EXECUTOR.format(
                    python_code=exec_data_processing_op_command,
                    python_main=exec_data_processing_op_args.strip(),
                ),
            ],
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
        ),
    ]

    # Format each string in the args list of each init container
    for container in init_containers:
        if container.name == "sdg-op-fetch-taxonomy-data":
            container.args = [
                arg.format(
                    exec_git_clone_op_repo_url=exec_git_clone_op_repo_url or "",
                    exec_git_clone_op_repo_branch=exec_git_clone_op_repo_branch or "",
                    exec_git_clone_op_repo_pr=exec_git_clone_op_repo_pr or "",
                    TAXONOMY_PATH=TAXONOMY_PATH,
                )
                for arg in container.args
            ]

    container = kubernetes.client.V1Container(
        name="copy-model-to-pvc",
        image=TOOLBOX_IMAGE,
        command=["/bin/sh", "-c"],
        args=[
            f"cp -r -v {DATA_PVC_MOUNT_PATH} {DATA_PVC_MOUNT_PATH}"
        ],  # TODO: fix me, dumb line to pass linter, this feat is unused anyway
        volume_mounts=get_vol_mount(),
    )

    volumes = get_vol()

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "sdg"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            init_containers=init_containers,
            containers=[container],
            volumes=volumes,
        ),
    )

    # Create the specification of deployment
    spec = kubernetes.client.V1JobSpec(
        template=template,
    )

    # Instantiate the job object
    job = kubernetes.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=kubernetes.client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=spec,
    )

    return job


def create_sdg_data_fetch_job(
    namespace: str,
    job_name: str,
    sdg_object_store_secret: str,
    force_pull: bool = False,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to fetch SDG data from an object store.

    Args:
        namespace (str): The namespace in which the job will be created.
        job_name (str): The name of the job.
        sdg_object_store_secret (str): The name of the Kubernetes Secret containing the SDG object store credentials.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """

    exec_data_processing_op_command = """
from typing import *

def data_processing_op(
    sdg: str,
    processed_data: str,
    model: str,
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
        model_path=model,
        data_path=f"{sdg}/*_train_msgs*.jsonl",
        data_output_dir=processed_data,
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
"""
    exec_data_processing_op_args = """
data_processing_op(max_seq_len=4096, max_batch_len=20000, sdg="/data/data", model="/data/model", processed_data="/data/processed_data")
"""

    init_containers = [
        kubernetes.client.V1Container(
            name="fetch-sdg-files-from-object-store",
            image=DS_IMAGE,
            command=["/bin/sh", "-c"],
            args=[
                DATA_SCRIPT.format(
                    strategy="download",
                    force_pull=force_pull,
                    data_pvc_mount_path=DATA_PVC_MOUNT_PATH,
                )
            ],
            volume_mounts=get_vol_mount(),
            env=[
                kubernetes.client.V1EnvVar(
                    name="SDG_OBJECT_STORE_ENDPOINT",
                    value_from=kubernetes.client.V1EnvVarSource(
                        secret_key_ref=kubernetes.client.V1SecretKeySelector(
                            name=sdg_object_store_secret, key="endpoint", optional=True
                        )
                    ),
                ),
                kubernetes.client.V1EnvVar(
                    name="SDG_OBJECT_STORE_BUCKET",
                    value_from=kubernetes.client.V1EnvVarSource(
                        secret_key_ref=kubernetes.client.V1SecretKeySelector(
                            name=sdg_object_store_secret, key="bucket", optional=False
                        )
                    ),
                ),
                kubernetes.client.V1EnvVar(
                    name="SDG_OBJECT_STORE_ACCESS_KEY",
                    value_from=kubernetes.client.V1EnvVarSource(
                        secret_key_ref=kubernetes.client.V1SecretKeySelector(
                            name=sdg_object_store_secret,
                            key="access_key",
                            optional=False,
                        )
                    ),
                ),
                kubernetes.client.V1EnvVar(
                    name="SDG_OBJECT_STORE_SECRET_KEY",
                    value_from=kubernetes.client.V1EnvVarSource(
                        secret_key_ref=kubernetes.client.V1SecretKeySelector(
                            name=sdg_object_store_secret,
                            key="secret_key",
                            optional=False,
                        )
                    ),
                ),
                kubernetes.client.V1EnvVar(
                    name="SDG_OBJECT_STORE_REGION",
                    value_from=kubernetes.client.V1EnvVarSource(
                        secret_key_ref=kubernetes.client.V1SecretKeySelector(
                            name=sdg_object_store_secret, key="region", optional=True
                        )
                    ),
                ),
                kubernetes.client.V1EnvVar(
                    name="SDG_OBJECT_STORE_DATA_KEY",
                    value_from=kubernetes.client.V1EnvVarSource(
                        secret_key_ref=kubernetes.client.V1SecretKeySelector(
                            name=sdg_object_store_secret, key="data_key", optional=False
                        )
                    ),
                ),
                kubernetes.client.V1EnvVar(
                    name="SDG_OBJECT_STORE_VERIFY_TLS",
                    value_from=kubernetes.client.V1EnvVarSource(
                        secret_key_ref=kubernetes.client.V1SecretKeySelector(
                            name=sdg_object_store_secret,
                            key="verify_tls",
                            optional=True,
                        )
                    ),
                ),
            ],
        )
    ]

    container = kubernetes.client.V1Container(
        name="sdg-preprocess",
        image=RHELAI_IMAGE,
        command=["/bin/sh", "-ce"],
        args=[
            PYTHON_EXECUTOR.format(
                python_code=exec_data_processing_op_command,
                python_main=exec_data_processing_op_args.strip(),
            ),
        ],
        volume_mounts=get_vol_mount(),
        security_context=get_security_context(),
        env_from=[
            kubernetes.client.V1EnvFromSource(
                config_map_ref=kubernetes.client.V1ConfigMapEnvSource(name=K8S_NAME)
            ),
            kubernetes.client.V1EnvFromSource(
                secret_ref=kubernetes.client.V1SecretEnvSource(name=K8S_NAME)
            ),
        ],
    )

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "sdg-data-fetch"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            init_containers=init_containers,
            containers=[container],
            volumes=get_vol(),
        ),
    )

    # Create the specification of deployment
    spec = kubernetes.client.V1JobSpec(
        template=template,
    )

    # Instantiate the job object
    job = kubernetes.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=kubernetes.client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=spec,
    )

    return job


def create_eval_job(
    namespace: str,
    job_name: str,
    eval_type: str,
    nproc_per_node: int = 1,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to run Evaluation steps.

    Args:
        namespace (str): The namespace in which the job will be created.
        job_name (str): The name of the job.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """

    # if eval_type == "mmlu":
    #     init_containers = [
    #         kubernetes.client.V1Container(
    #             name=f"run-eval-{eval_type}",
    #             image="",
    #             command=,
    #             args=,
    #             volume_mounts=[
    #                 kubernetes.client.V1VolumeMount(
    #                     name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
    #                 ),
    #             ],
    #         )
    #     ]
    #     container = kubernetes.client.V1Container(
    #         name=f"output-eval-{eval_type}-scores",
    #         image="",
    #         command=["/bin/sh", "-c"],
    #         args=[f"cat {MMLU_SCORES_PATH}"],
    #         volume_mounts=[
    #             kubernetes.client.V1VolumeMount(
    #                 name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
    #             ),
    #         ],
    #     )

    exec_run_mt_bench_op_command = """
from typing import *

def run_mt_bench_op(
    models_path_prefix: str,
    mt_bench_output: str,
    merge_system_user_message: bool,
    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - 'auto'
    # with 'auto', number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    max_workers: str,
    models_list: List[str] = None,
    models_folder: Optional[str] = None,
    device: str = None,
    best_score_file: Optional[str] = None,
) -> NamedTuple("outputs", best_model=str, best_score=float):
    import json
    import os

    import torch
    from instructlab.eval.mt_bench import MTBenchEvaluator

    VLLM_SERVER = "http://localhost:8000/v1"

    def launch_vllm(
        model_path: str, gpu_count: int, retries: int = 120, delay: int = 10
    ):
        import subprocess
        import sys
        import time

        import requests

        if gpu_count > 0:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
                "--tensor-parallel-size",
                str(gpu_count),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
            ]

        subprocess.Popen(args=command)

        print(f"Waiting for vLLM server to start at {VLLM_SERVER}...")

        for attempt in range(retries):
            try:
                response = requests.get(f"{VLLM_SERVER}/models")
                if response.status_code == 200:
                    print(f"vLLM server is up and running at {VLLM_SERVER}.")
                    return
            except requests.ConnectionError:
                pass

            print(
                f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Failed to start vLLM server at {VLLM_SERVER} after {retries} retries."
        )

    # This seems like excessive effort to stop the vllm process, but merely saving & killing the pid doesn't work
    # Also, the base image does not include 'pkill' cmd, so can't pkill -f vllm.entrypoints.openai.api_server either
    def stop_vllm():
        import psutil

        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            cmdline = process.info.get("cmdline")
            if cmdline and "vllm.entrypoints.openai.api_server" in cmdline:
                print(
                    f"Found vLLM server process with PID: {process.info['pid']}, terminating..."
                )
                try:
                    process.terminate()  # Try graceful termination
                    process.wait(timeout=5)  # Wait a bit for it to terminate
                    if process.is_running():
                        print(
                            f"Forcefully killing vLLM server process with PID: {process.info['pid']}"
                        )
                        process.kill()  # Force kill if it's still running
                    print(
                        f"Successfully stopped vLLM server with PID: {process.info['pid']}"
                    )
                except psutil.NoSuchProcess:
                    print(f"Process with PID {process.info['pid']} no longer exists.")
                except psutil.AccessDenied:
                    print(
                        f"Access denied when trying to terminate process with PID {process.info['pid']}."
                    )
                except Exception as e:
                    print(
                        f"Failed to terminate process with PID {process.info['pid']}. Error: {e}"
                    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    gpu_available = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if gpu_available
        else "No GPU available"
    )
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    print(f"GPU Available: {gpu_available}, {gpu_name}")

    if models_list is None and models_folder:
        models_list = os.listdir(models_folder)

    judge_api_key = os.getenv("JUDGE_API_KEY", "")
    judge_model_name = os.getenv("JUDGE_NAME")
    judge_endpoint = os.getenv("JUDGE_ENDPOINT")

    scores = {}
    all_mt_bench_data = []

    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - 'auto'
    # with 'auto', number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    if max_workers == "auto":
        try:
            usable_cpu_count = len(os.sched_getaffinity(0)) // 2
        except AttributeError:
            usable_cpu_count = multiprocessing.cpu_count() // 2
        max_workers = usable_cpu_count

    for model_name in models_list:
        print(f"Serving candidate model: {model_name}")
        model_path = f"{models_path_prefix}/{model_name}"

        launch_vllm(model_path, gpu_count)

        # model ID is the model_path value in vLLM
        evaluator = MTBenchEvaluator(
            model_name=model_path,
            judge_model_name=judge_model_name,
            output_dir="/tmp/eval_output",
            merge_system_user_message=merge_system_user_message,
        )

        evaluator.gen_answers(
            server_url=VLLM_SERVER,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        stop_vllm()

        overall_score, qa_pairs, turn_scores, error_rate = evaluator.judge_answers(
            server_url=judge_endpoint,
            api_key=judge_api_key,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        mt_bench_data = {
            "report_title": "SKILLS EVALUATION REPORT",
            "model": model_path,
            "judge_model": judge_model_name,
            "overall_score": overall_score,
            "turn_scores": turn_scores,
            "qa_scores": qa_pairs,
            "error_rate": error_rate,
        }

        all_mt_bench_data.append(mt_bench_data)
        scores[model_path] = overall_score

    with open(mt_bench_output, "w", encoding="utf-8") as f:
        json.dump(all_mt_bench_data, f, indent=4)

    outputs = NamedTuple("outputs", best_model=str, best_score=float)
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    if best_score_file:
        with open(best_score_file, "w", encoding="utf-8") as f:
            json.dump({"best_model": best_model, "best_score": best_score}, f, indent=4)

    # Rename the best model directory to "candidate_model" for the next step
    # So we know which model to use for the final evaluation
    os.rename(
        os.path.join(models_path_prefix, best_model),
        os.path.join(models_path_prefix, "candidate_model"),
    )

    return outputs(best_model=best_model, best_score=best_score)
"""
    exec_run_mt_bench_op_args = """
run_mt_bench_op(best_score_file="/data/mt-bench-best.txt",mt_bench_output="/data/mt-bench-results.txt", models_folder="/data/model/output/hf_format", models_path_prefix="/data/model/output/hf_format", max_workers="auto", merge_system_user_message=False)
"""
    exec_run_mt_bench_branch_op_command = """
from typing import *

def run_mt_bench_op(
    models_path_prefix: str,
    mt_bench_output: str,
    merge_system_user_message: bool,
    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - 'auto'
    # with 'auto', number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    max_workers: str,
    models_list: List[str] = None,
    models_folder: Optional[str] = None,
    device: str = None,
    best_score_file: Optional[str] = None,
) -> NamedTuple("outputs", best_model=str, best_score=float):
    import json
    import os

    import torch
    from instructlab.eval.mt_bench import MTBenchEvaluator

    VLLM_SERVER = "http://localhost:8000/v1"

    def launch_vllm(
        model_path: str, gpu_count: int, retries: int = 120, delay: int = 10
    ):
        import subprocess
        import sys
        import time

        import requests

        if gpu_count > 0:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
                "--tensor-parallel-size",
                str(gpu_count),
            ]
        else:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                model_path,
            ]

        subprocess.Popen(args=command)

        print(f"Waiting for vLLM server to start at {VLLM_SERVER}...")

        for attempt in range(retries):
            try:
                response = requests.get(f"{VLLM_SERVER}/models")
                if response.status_code == 200:
                    print(f"vLLM server is up and running at {VLLM_SERVER}.")
                    return
            except requests.ConnectionError:
                pass

            print(
                f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."
            )
            time.sleep(delay)

        raise RuntimeError(
            f"Failed to start vLLM server at {VLLM_SERVER} after {retries} retries."
        )

    # This seems like excessive effort to stop the vllm process, but merely saving & killing the pid doesn't work
    # Also, the base image does not include 'pkill' cmd, so can't pkill -f vllm.entrypoints.openai.api_server either
    def stop_vllm():
        import psutil

        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
            cmdline = process.info.get("cmdline")
            if cmdline and "vllm.entrypoints.openai.api_server" in cmdline:
                print(
                    f"Found vLLM server process with PID: {process.info['pid']}, terminating..."
                )
                try:
                    process.terminate()  # Try graceful termination
                    process.wait(timeout=5)  # Wait a bit for it to terminate
                    if process.is_running():
                        print(
                            f"Forcefully killing vLLM server process with PID: {process.info['pid']}"
                        )
                        process.kill()  # Force kill if it's still running
                    print(
                        f"Successfully stopped vLLM server with PID: {process.info['pid']}"
                    )
                except psutil.NoSuchProcess:
                    print(f"Process with PID {process.info['pid']} no longer exists.")
                except psutil.AccessDenied:
                    print(
                        f"Access denied when trying to terminate process with PID {process.info['pid']}."
                    )
                except Exception as e:
                    print(
                        f"Failed to terminate process with PID {process.info['pid']}. Error: {e}"
                    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    gpu_available = torch.cuda.is_available()
    gpu_name = (
        torch.cuda.get_device_name(torch.cuda.current_device())
        if gpu_available
        else "No GPU available"
    )
    gpu_count = torch.cuda.device_count() if gpu_available else 0

    print(f"GPU Available: {gpu_available}, {gpu_name}")

    if models_list is None and models_folder:
        models_list = os.listdir(models_folder)

    judge_api_key = os.getenv("JUDGE_API_KEY", "")
    judge_model_name = os.getenv("JUDGE_NAME")
    judge_endpoint = os.getenv("JUDGE_ENDPOINT")

    scores = {}
    all_mt_bench_data = []

    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - 'auto'
    # with 'auto', number of gpus allocated for serving is calculated based on environment
    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36
    if max_workers == "auto":
        try:
            usable_cpu_count = len(os.sched_getaffinity(0)) // 2
        except AttributeError:
            usable_cpu_count = multiprocessing.cpu_count() // 2
        max_workers = usable_cpu_count

    for model_name in models_list:
        print(f"Serving candidate model: {model_name}")
        model_path = f"{models_path_prefix}/{model_name}"

        launch_vllm(model_path, gpu_count)

        # model ID is the model_path value in vLLM
        evaluator = MTBenchEvaluator(
            model_name=model_path,
            judge_model_name=judge_model_name,
            output_dir="/tmp/eval_output",
            merge_system_user_message=merge_system_user_message,
        )

        evaluator.gen_answers(
            server_url=VLLM_SERVER,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        stop_vllm()

        overall_score, qa_pairs, turn_scores, error_rate = evaluator.judge_answers(
            server_url=judge_endpoint,
            api_key=judge_api_key,
            serving_gpus=gpu_count,
            max_workers=max_workers,
        )

        mt_bench_data = {
            "report_title": "SKILLS EVALUATION REPORT",
            "model": model_path,
            "judge_model": judge_model_name,
            "overall_score": overall_score,
            "turn_scores": turn_scores,
            "qa_scores": qa_pairs,
            "error_rate": error_rate,
        }

        all_mt_bench_data.append(mt_bench_data)
        scores[model_path] = overall_score

    with open(mt_bench_output, "w", encoding="utf-8") as f:
        json.dump(all_mt_bench_data, f, indent=4)

    outputs = NamedTuple("outputs", best_model=str, best_score=float)
    best_model = max(scores, key=scores.get)
    best_score = scores[best_model]
    if best_score_file:
        with open(best_score_file, "w", encoding="utf-8") as f:
            json.dump({"best_model": best_model, "best_score": best_score}, f, indent=4)

    # Rename the best model directory to "candidate_model" for the next step
    # So we know which model to use for the final evaluation
    os.rename(
        os.path.join(models_path_prefix, best_model),
        os.path.join(models_path_prefix, "candidate_model"),
    )

    return outputs(best_model=best_model, best_score=best_score)
"""
    exec_run_mt_bench_branch_op_args = """
run_mt_bench_op(best_score_file="/data/mt-bench-best.txt",mt_bench_output="/data/mt-bench-results.txt", models_folder="/data/model/output/hf_format", models_path_prefix="/data/model/output/hf_format", max_workers="auto", merge_system_user_message=False)
"""

    if eval_type == "mt-bench":
        init_containers = [
            kubernetes.client.V1Container(
                name=f"run-eval-{eval_type}",
                image=RHELAI_IMAGE,
                command=["/bin/sh", "-ce"],
                args=[
                    PYTHON_EXECUTOR.format(
                        python_code=exec_run_mt_bench_op_command,
                        python_main=exec_run_mt_bench_op_args.strip(),
                    ),
                ],
                volume_mounts=get_vol_mount(),
                security_context=get_security_context(),
                env_from=[
                    kubernetes.client.V1EnvFromSource(
                        secret_ref=kubernetes.client.V1SecretEnvSource(
                            name=JUDGE_SERVING_NAME
                        )
                    ),
                ],
                resources=kubernetes.client.V1ResourceRequirements(
                    requests={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
                    limits={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
                ),
            )
        ]
        container = kubernetes.client.V1Container(
            name=f"output-eval-{eval_type}-scores",
            image=RHELAI_IMAGE,
            command=["/bin/sh", "-c"],
            args=[f"cat {MT_BENCH_SCORES_PATH}"],
            security_context=get_security_context(),
            volume_mounts=get_vol_mount(),
        )
    elif eval_type == "mt-bench-branch":
        init_containers = [
            kubernetes.client.V1Container(
                name=f"run-eval-{eval_type}",
                image=RHELAI_IMAGE,
                command=["/bin/sh", "-ce"],
                args=[
                    PYTHON_EXECUTOR.format(
                        python_code=exec_run_mt_bench_branch_op_command,
                        python_main=exec_run_mt_bench_branch_op_args.strip(),
                    ),
                ],
                volume_mounts=get_vol_mount(),
                security_context=get_security_context(),
                env_from=[
                    kubernetes.client.V1EnvFromSource(
                        secret_ref=kubernetes.client.V1SecretEnvSource(
                            name=JUDGE_SERVING_NAME
                        )
                    ),
                ],
                resources=kubernetes.client.V1ResourceRequirements(
                    requests={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
                    limits={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
                ),
            )
        ]
        container = kubernetes.client.V1Container(
            name=f"output-eval-{eval_type}-scores",
            image=RHELAI_IMAGE,
            command=["/bin/sh", "-c"],
            args=[f"cat {MT_BENCH_BRANCH_SCORES_PATH}"],
            security_context=get_security_context(),
            volume_mounts=get_vol_mount(),
        )
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": f"eval-{eval_type}"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            init_containers=init_containers,
            containers=[container],
            volumes=get_vol(),
        ),
    )

    # Create the specification of deployment
    spec = kubernetes.client.V1JobSpec(
        template=template,
    )

    # Instantiate the job object
    job = kubernetes.client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=kubernetes.client.V1ObjectMeta(name=job_name, namespace=namespace),
        spec=spec,
    )

    return job


def run_job(namespace: str, job: kubernetes.client.V1Job) -> str:
    """
    Create and run a Kubernetes job in the specified namespace, and wait for its completion.

    Args:
        namespace (str): The namespace in which to create the job.
        job (kubernetes.client.V1Job): The job object to be created and run.

    Returns:
        str: The last container's logs.

    Prints:
        str: The status of the job during its execution.

    The function will print the job's status as it progresses and will stop watching once the job
    either succeeds or fails. If the job fails, it will also print the logs of the failed pod.
    """
    # Create a job
    batch_v1 = kubernetes.client.BatchV1Api()
    core_v1 = kubernetes.client.CoreV1Api()
    try:
        resp = batch_v1.create_namespaced_job(body=job, namespace=namespace)
        logger.info("Job created '%s/%s'", namespace, resp.metadata.name)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            logger.info(
                "%s '%s/%s' already exists.",
                job.kind,
                namespace,
                job.metadata.name,
            )
        else:
            raise

    # Wait for the job to complete
    w = kubernetes.watch.Watch()
    pod_log = None
    for event in w.stream(batch_v1.list_namespaced_job, namespace=namespace):
        job_event = event["object"]
        if job_event.metadata.name != job.metadata.name:
            continue
        logger.info("Job: %s - %s", job.metadata.name, job_event.status)
        if job_event.status.succeeded == 1:
            logger.info("Job completed successfully.")
            pods = core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector="app={}".format(
                    job.spec.template.metadata.labels["app"]
                ),
            )
            # On success return the logs of the last pod which contains the output
            # (useful to get eval scores)
            pod_log = core_v1.read_namespaced_pod_log(
                name=pods.items[0].metadata.name, namespace=namespace
            )
            w.stop()
        elif job_event.status.failed == 1:
            logger.error("Job failed. Pod logs:")
            pods = core_v1.list_namespaced_pod(
                namespace=namespace,
                label_selector="app={}".format(
                    job.spec.template.metadata.labels["app"]
                ),
            )
            for pod in pods.items:

                def log_pod_containers(pod, container_type):
                    containers = getattr(pod.spec, container_type)
                    # If there are no containers, skip (e.g. noinit_containers)
                    if containers is None:
                        return
                    for container in containers:
                        try:
                            pod_log = core_v1.read_namespaced_pod_log(
                                name=pod.metadata.name,
                                namespace=namespace,
                                container=container.name,
                            )
                            logger.error(
                                "Logs for pod %s, %s %s:\n%s",
                                pod.metadata.name,
                                container_type[:-1],  # Remove the trailing 's'
                                container.name,
                                pod_log,
                            )
                        except kubernetes.client.rest.ApiException as exc:
                            if exc.status == 400:
                                continue

                log_pod_containers(pod, "init_containers")
                log_pod_containers(pod, "containers")
            w.stop()
            raise RuntimeError("Job failed.")

    return pod_log


def create_pvc(
    name: str,
    namespace: str,
    storage_class: str,
    access_modes: list,
    size: str,
) -> kubernetes.client.V1PersistentVolumeClaim:
    """
    Create a PersistentVolumeClaim (PVC) in the specified namespace.

    Args:
        namespace (str): The namespace in which to create the PVC.
        storage_class (str): The storage class for the PVC.
        access_modes (list): The access modes for the PVC.
        size (str): The size of the PVC.

    Returns:
        kubernetes.client.V1PersistentVolumeClaim: The created PVC object.
    """
    # Create a PVC
    return kubernetes.client.V1PersistentVolumeClaim(
        metadata=kubernetes.client.V1ObjectMeta(name=name, namespace=namespace),
        spec=kubernetes.client.V1PersistentVolumeClaimSpec(
            access_modes=access_modes,
            storage_class_name=storage_class,
            resources=kubernetes.client.V1ResourceRequirements(
                requests={"storage": size}
            ),
        ),
    )


@run.command(name="sdg")
@click.pass_context
def sdg(
    ctx: click.Context,
) -> None:
    """
    Preprocesses SDG data by creating a Persistent Volume Claim (PVC) and
    initiating a job to run a pod for SDG data preprocessing.

    Steps:
        1. Creates a PVC to hold SDG data and transformed SDG data.
        2. Initiates a job to run a pod for SDG data preprocessing.
    """
    # Populate variables from context
    namespace = ctx.obj["namespace"]
    taxonomy_repo_url = ctx.obj["taxonomy_repo_url"]
    taxonomy_repo_branch = ctx.obj["taxonomy_repo_branch"]
    taxonomy_repo_pr = ctx.obj["taxonomy_repo_pr"]
    storage_class = ctx.obj["storage_class"]
    serving_endpoint = ctx.obj["serving_endpoint"]
    serving_model = ctx.obj["serving_model"]

    # check in the context
    if not taxonomy_repo_branch and not taxonomy_repo_pr:
        raise ValueError(
            "Either '--taxonomy-repo-branch' or '--taxonomy-repo-pr' must be provided to the 'run' command."
        )

    logger.info("Running setup for SDG.")
    # Request the Kubernetes API
    v1 = kubernetes.client.CoreV1Api()

    # list of PVCs to create and their details
    pvcs = [
        {
            "name": DATA_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteOnce"],
            "size": "200Gi",
        },
    ]
    for pvc in pvcs:
        try:
            v1.create_namespaced_persistent_volume_claim(
                namespace=namespace, body=create_pvc(**pvc)
            )
            logger.info("Successfully created PVC '%s' created.", pvc.get("name"))
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("PVC '%s' already exists.", pvc["name"])
            else:
                raise

    # Create SDG config map/secret with api_key, serving endpoint
    cms = list(yaml.safe_load_all(KFP_MODEL_SERVER_CM))
    for cm in cms:
        try:
            # if this is a ConfigMap
            if cm["kind"] == "ConfigMap":
                if serving_endpoint:
                    cm["data"]["endpoint"] = serving_endpoint
                if serving_model:
                    cm["data"]["model"] = serving_model
                v1.create_namespaced_config_map(namespace=namespace, body=cm)
                logger.info("Successfully created ConfigMap '%s' created.", cm)
            elif cm["kind"] == "Secret":
                # if this is a Secret
                v1.create_namespaced_secret(namespace=namespace, body=cm)
                logger.info("Successfully created Secret '%s' created.", cm)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info(
                    "%s '%s' already exists.", cm["kind"], cm["metadata"]["name"]
                )
            else:
                raise

    # Create the job to run the pod to execute the SDG data preprocessing
    # Example usage
    job = create_sdg_job(
        namespace=namespace,
        job_name="sdg",
        exec_git_clone_op_repo_url=taxonomy_repo_url,
        exec_git_clone_op_repo_branch=taxonomy_repo_branch,
        exec_git_clone_op_repo_pr=taxonomy_repo_pr,
    )
    run_job(namespace, job)
    logger.info("SDG setup completed.")


def validate_url(url: str) -> str:
    """
    Validate if the given string is a valid URL.

    Args:
        url (str): The URL string to validate.

    Returns:
        str: The original URL if valid.

    Raises:
        ValueError: If the URL is not valid.
    """
    parsed = urlparse(url)
    if not all([parsed.scheme, parsed.netloc]):
        raise ValueError(f"Invalid URL: {url}")
    return url


@run.command(name="sdg-data-fetch")
@click.pass_context
def sdg_data_fetch(
    ctx: click.Context,
) -> None:
    """
    Fetches SDG data from an object store and put in a Persistent Volume Claim (PVC)
    """
    # Populate variables from context
    namespace = ctx.obj["namespace"]
    storage_class = ctx.obj["storage_class"]
    judge_serving_endpoint = ctx.obj["judge_serving_endpoint"]
    judge_serving_model_name = ctx.obj["judge_serving_model_name"]
    judge_serving_model_api_key = ctx.obj["judge_serving_model_api_key"]
    sdg_object_store_endpoint = ctx.obj["sdg_object_store_endpoint"]
    sdg_object_store_bucket = ctx.obj["sdg_object_store_bucket"]
    sdg_object_store_access_key = ctx.obj["sdg_object_store_access_key"]
    sdg_object_store_secret_key = ctx.obj["sdg_object_store_secret_key"]
    sdg_object_store_region = ctx.obj["sdg_object_store_region"]
    sdg_object_store_data_key = ctx.obj["sdg_object_store_data_key"]
    sdg_object_store_verify_tls = ctx.obj["sdg_object_store_verify_tls"]
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]
    force_pull = ctx.obj["force_pull"]

    # Make sure the endpoint is a valid URL
    validate_url(judge_serving_endpoint)

    # Check if all required arguments are provided
    if not sdg_object_store_secret:
        if not all(
            [
                sdg_object_store_bucket,
                sdg_object_store_access_key,
                sdg_object_store_secret_key,
                sdg_object_store_data_key,
            ]
        ):
            # Endpoint is optional if AWS S3 is used
            raise ValueError(
                "All of '--sdg-object-store-bucket', "
                "'--sdg-object-store-access-key', '--sdg-object-store-secret-key', '--sdg-object-store-data-key' "
                "must be provided to the 'sdg-data-fetch' command. Alternatively, provide "
                "'--sdg-object-store-secret' to use a Kubernetes Secret."
            )

    logger.info("Running setup for SDG data fetch.")

    # Request the Kubernetes API
    v1 = kubernetes.client.CoreV1Api()

    # Create the object store secret if it does not exist
    if (
        # Endpoint (if AWS S3 is used) and Region are optional
        all(
            [
                sdg_object_store_bucket,
                sdg_object_store_access_key,
                sdg_object_store_secret_key,
                sdg_object_store_data_key,
            ]
        )
        and not sdg_object_store_secret
    ):
        sdg_object_store_secret = SDG_OBJECT_STORE_SECRET_NAME
        secret = kubernetes.client.V1Secret(
            metadata=kubernetes.client.V1ObjectMeta(
                name=sdg_object_store_secret, namespace=namespace
            ),
            string_data={
                "bucket": sdg_object_store_bucket,
                "access_key": sdg_object_store_access_key,
                "secret_key": sdg_object_store_secret_key,
                "data_key": sdg_object_store_data_key,
            },
        )

        # Endpoint is optional if AWS S3 is used
        if sdg_object_store_endpoint:
            validate_url(sdg_object_store_endpoint)
            secret.string_data["endpoint"] = sdg_object_store_endpoint

        # Region is optional
        if sdg_object_store_region:
            secret.string_data["region"] = sdg_object_store_region

        if not sdg_object_store_verify_tls:
            secret.string_data["verify_tls"] = "false"

        try:
            v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret exists, verify the presence of the keys
    elif sdg_object_store_secret:
        secret = v1.read_namespaced_secret(
            name=sdg_object_store_secret, namespace=namespace
        )

        def decode_base64(data):
            return base64.b64decode(data).decode("utf-8")

        if secret.data.get("endpoint"):
            endpoint = decode_base64(secret.data.get("endpoint"))
            validate_url(endpoint)

        if not all(
            [
                secret.data.get("bucket"),
                secret.data.get("access_key"),
                secret.data.get("secret_key"),
                secret.data.get("data_key"),
            ]
        ):
            raise ValueError(
                f"The provided secret {sdg_object_store_secret} must contain the keys:"
                "'bucket', 'access_key', 'secret_key', 'data_key'.",
            )

    # Create Secret config details for evaluation
    judge_serving_details_secret = JUDGE_SERVING_NAME
    secret = kubernetes.client.V1Secret(
        metadata=kubernetes.client.V1ObjectMeta(
            name=judge_serving_details_secret, namespace=namespace
        ),
        string_data={
            "JUDGE_NAME": judge_serving_model_name,
            "JUDGE_API_KEY": judge_serving_model_api_key,
            "JUDGE_ENDPOINT": judge_serving_endpoint,
        },
    )

    try:
        v1.create_namespaced_secret(namespace=namespace, body=secret)
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            logger.info("Secret '%s' already exists.", secret.metadata.name)
        else:
            raise

    # list of PVCs to create and their details
    pvcs = [
        {
            "name": DATA_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteMany"],
            "size": "200Gi",  # Allocate size for a few models and large SDG data sets
        },
    ]
    for pvc in pvcs:
        try:
            v1.create_namespaced_persistent_volume_claim(
                namespace=namespace, body=create_pvc(**pvc)
            )
            logger.info("Successfully creayed PVC '%s' created.", pvc.get("name"))
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("PVC '%s' already exists.", pvc["name"])
            else:
                raise

    # Create the job to run the pod to execute the SDG data fetch
    job = create_sdg_data_fetch_job(
        namespace=namespace,
        job_name="sdg-data-fetch",
        sdg_object_store_secret=sdg_object_store_secret,
        force_pull=force_pull,
    )

    # Run the job
    run_job(namespace, job)
    logger.info("SDG fetch completed.")


@run.command(name="train")
@click.pass_context
def train(
    ctx: click.Context,
) -> None:
    """
    Run the distributed training.
    """
    namespace = ctx.obj["namespace"]
    training_phase = ctx.obj["training_phase"]
    path_to_model = ctx.obj["model_to_train"]
    nproc_per_node: int = ctx.obj["nproc_per_node"]
    training_1_epoch_num: int = ctx.obj["training_1_epoch_num"]
    training_2_epoch_num: int = ctx.obj["training_2_epoch_num"]

    if training_phase is None:
        raise ValueError("Training phase must be provided with --training-phase=[1|2]")

    # During the initial training
    if path_to_model is None:
        path_to_model = DATA_PVC_MODEL_PATH

    epoch_num = None
    if training_phase == "1":
        epoch_num = training_1_epoch_num
    elif training_phase == "2":
        epoch_num = training_2_epoch_num

    logger.info("Running multi-phased distributed training phase %s", training_phase)
    worker_replicas = PYTORCH_NNODES - 1
    pytorch_training_job_yaml = yaml.safe_load(
        PYTORCH_TRAINING_JOB.format(
            name=f"train-phase-{training_phase}",
            data_pvc_name=DATA_PVC_NAME,
            path_to_model=path_to_model,
            nproc_per_node=nproc_per_node,
            nnodes=PYTORCH_NNODES,
            image=RHELAI_IMAGE,
            worker_replicas=worker_replicas,
            epoch_num=epoch_num,
            phase_num=training_phase,
        )
    )

    api = kubernetes.client.CustomObjectsApi()

    try:
        api.create_namespaced_custom_object(
            group="kubeflow.org",
            version="v1",
            namespace=namespace,
            plural="pytorchjobs",
            body=pytorch_training_job_yaml,
        )
    except kubernetes.client.rest.ApiException as exc:
        if exc.status == 409:
            logger.info(
                "%s '%s/%s' already exists.",
                pytorch_training_job_yaml["kind"],
                namespace,
                pytorch_training_job_yaml["metadata"]["name"],
            )
        else:
            raise

    # Get the CR status and wait for it to be completed
    w = kubernetes.watch.Watch()
    for event in w.stream(
        api.list_namespaced_custom_object,
        group="kubeflow.org",
        version="v1",
        namespace=namespace,
        plural="pytorchjobs",
    ):
        job_event = event["object"]
        if (
            job_event["metadata"]["name"]
            != pytorch_training_job_yaml["metadata"]["name"]
        ):
            continue
        job_name = job_event["metadata"]["name"]

        if "status" not in job_event or "conditions" not in job_event["status"]:
            continue
        logger.info(
            "Job: %s - %s",
            job_name,
            job_event["status"].get("conditions", "No conditions yet"),
        )

        # TODO: check pod status to exit if training pods are failing
        for condition in job_event["status"]["conditions"]:
            if condition["type"] == "Succeeded":
                logger.info(
                    "Job '%s' completed successfully: %s", job_name, condition["reason"]
                )
                w.stop()
            elif condition["type"] == "Failed":
                logger.error("Job' %s' failed: %s", job_name, condition["reason"])
                w.stop()
                raise RuntimeError("Job failed.")


@run.command(name="evaluation")
@click.pass_context
def evaluation(ctx: click.Context) -> str:
    """
    Run the evaluation phase and return the scores as a JSON string.

    Args:
        ctx (click.Context): The Click context object.
        eval_type (str): The type of evaluation to run.

    Returns:
        str: The evaluation scores as a JSON string.
    """
    namespace = ctx.obj["namespace"]
    eval_type = ctx.obj["eval_type"]

    if eval_type is None:
        raise ValueError("Evaluation type must be provided with --eval-type=[mt-bench]")

    logger.info("Running %s evaluation.", eval_type)

    # Create and run the evaluation job
    job = create_eval_job(
        namespace=namespace, job_name=f"eval-{eval_type}", eval_type=eval_type
    )
    scores = run_job(namespace, job)
    scores = scores.replace("'", '"')

    try:
        scores_data = json.loads(scores)
        if isinstance(scores_data, dict):
            scores = json.dumps(scores_data)
        else:
            raise ValueError("Unexpected format for scores data")
    except json.JSONDecodeError as e:
        logger.error("Failed to parse scores: %s", e)
        raise

    logger.info("Evaluation scores: %s", scores)
    return scores


if __name__ == "__main__":
    # Configs can be set in Configuration class directly or using helper utility
    try:
        kubernetes.config.load_kube_config()
    except kubernetes.config.ConfigException:
        logger.info("Failed to load kube config. Trying in-cluster config")
        kubernetes.config.load_incluster_config()

    cli()
