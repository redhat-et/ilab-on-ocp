#!/usr/bin/env python3
# pylint: disable=too-many-lines

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
import time
import typing
from os import path
from urllib.parse import urlparse

import click
import kubernetes
import kubernetes.client
import kubernetes.client.exceptions
import kubernetes.client.rest
import kubernetes.config
import kubernetes.utils
import kubernetes.watch
import urllib3.exceptions
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s %(asctime)s %(name)s:%(lineno)d: %(message)s",
)
logger = logging.getLogger(__name__)

# IMAGES
DS_IMAGE = "quay.io/opendatahub/workbench-images:jupyter-datascience-ubi9-python-3.11-20241004-609ffb8"  # pylint: disable=line-too-long
RHELAI_IMAGE = "registry.redhat.io/rhelai1/instructlab-nvidia-rhel9:1.2"

# SDG
DEFAULT_REPO_URL = "https://github.com/instructlab/taxonomy.git"
K8S_NAME = "kfp-model-server"
SDG_OBJECT_STORE_SECRET_NAME = "sdg-object-store-credentials"
REPO_GRANITE_7B_IMAGE = "ibm-granite/granite-7b-base"  # used by HF downloader

# SDG DATA PREPROCESSING (before doing training, data has to be converted)
MAX_SEQ_LEN = 4096
MAX_BATCH_LEN = 20000

# DATA
DATA_PVC_NAME = "data"
DATA_PVC_MOUNT_PATH = "/data"
DATA_PVC_SDG_PATH = path.join(DATA_PVC_MOUNT_PATH, "data")
DATA_PVC_MODEL_PATH = path.join(DATA_PVC_MOUNT_PATH, "model")
DATA_VOLUME_NAME = "data"
TAXONOMY_PATH = path.join(DATA_PVC_MOUNT_PATH, "taxonomy")
DATA_PVC_OUTPUT_PATH = path.join(DATA_PVC_MOUNT_PATH, "output")
DATA_PVC_OUTPUT_DATA_PATH = path.join(DATA_PVC_OUTPUT_PATH, "data")
PREPROCESSED_DATA_PATH = path.join(DATA_PVC_SDG_PATH, "processed_data")
MT_BENCH_OUTPUT_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-results.txt")
MT_BENCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-best.txt")
MT_BENCH_BRANCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-branch-best.txt")
MMLU_BRANCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mmlu-branch-best.txt")
CANDIDATE_MODEL_PATH = path.join(
    DATA_PVC_MOUNT_PATH, "model/output/phase_2/hf_format/candidate_model"
)
SDG_GENERATED_DATA_PATH = path.join(DATA_PVC_MOUNT_PATH, "generated")
TAXONOMY_DATA_PATH = path.join(DATA_PVC_MOUNT_PATH, "taxonomy")
# MMLU_SCORES_PATH = "/output/mmlu-results.txt" - after training phase 1 is done MMLU is not performed anymore

# TRAINING
PYTORCH_NNODES = 2

# EVALUATION
EVAL_TYPE_MT_BENCH = "mt-bench"
EVAL_TYPE_FINAL = "final"
JUDGE_SERVING_NAME = "judge-serving-details"
MODEL_DTYPE = "bfloat16"
MAX_WORKERS = "auto"
MERGE_SYSTEM_USER_MESSAGE = False
FEW_SHOTS = 5
BATCH_SIZE = 8

# TEMPLATES
KFP_MODEL_SERVER_CM = """
# TODO: remove the following line and replace it with the actual ConfigMap/Secret
{{kfp_model_server_cm}}
"""

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
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/phase_1/hf_format/$(ls --sort=time {path_to_model}/output/phase_1/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  mkdir -p {data_pvc_model_path};
                  mkdir -p {data_pvc_sdg_path};
                  mkdir -p {path_to_model}/output/phase_{phase_num}
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path=/data/processed_data/data.jsonl \
                    --output_dir={path_to_model}/output/phase_{phase_num} \
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
                    --cpu_offload_params \
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
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/phase_1/hf_format/$(ls --sort=time {path_to_model}/output/phase_1/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  tmp_model=$(mktemp -d)
                  mkdir -p "$tmp_model";
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path=/data/processed_data/data.jsonl \
                    --output_dir="$tmp_model" \
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
                    --cpu_offload_params \
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

DATA_SCRIPT = """
set -e

export STRATEGY={strategy}

if [ -z "$STRATEGY" ] || [ "$STRATEGY" == "None" ]; then
    echo "STRATEGY is not set - must be 'download' or 'upload'"
    exit 1
fi

if [ "$STRATEGY" == "download" ]; then
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

    if python3 -c 'import boto3'; then
        echo 'boto3 is already installed'
    else
        if ! [ -x "$(command -v pip)" ]; then
            python3 -m ensurepip || python3 -m ensurepip --user || dnf install python3-pip -y
        fi
        python3 -m pip install boto3
    fi
fi

if [ "$STRATEGY" == "upload" ]; then
    export FINAL_DATA_TAR_FILE="$(date +"%Y-%m-%d_%H-%M-%S").$SDG_OBJECT_STORE_DATA_KEY"
    export FINAL_DATA_TAR_PATH="{data_pvc_mount_path}/$FINAL_DATA_TAR_FILE"
    echo "Final data tarball path: $FINAL_DATA_TAR_PATH"
    echo "Final data tarball file: $FINAL_DATA_TAR_FILE"
    echo "Archiving data before pushing to the object store"
    # Use '--ignore-failed-read' to ignore missing files, needed when no MMLU tasks directories are found MMLU_branch is skipped
    # So '{mmlu_branch_scores_path}' will not exist
    tar --create \
      --gzip \
      --verbose \
      --ignore-failed-read \
      --file "$FINAL_DATA_TAR_PATH" {mt_bench_output_path} {mt_bench_scores_path} {mt_bench_branch_scores_path} {mmlu_branch_scores_path} {candidate_model_path}
fi

tmp=$(mktemp -d)
cat <<EOF > "$tmp"/s3.py
import os
import boto3
import sys
import threading

# Credit: https://gist.github.com/egeulgen/538aadc90275d79d514a5bacc4d5694e
class ProgressPercentage(object):
    ''' Progress Class
    Class for calculating and displaying download progress
    '''
    def __init__(self, client, bucket, filename):
        ''' Initialize
        initialize with: file name, file size and lock.
        Set seen_so_far to 0. Set progress bar length
        '''
        self._filename = filename
        self._size = float(os.path.getsize(filename)) if os.getenv('STRATEGY') == 'upload' else client.head_object(Bucket=bucket, Key=filename)['ContentLength']
        self._seen_so_far = 0
        self._lock = threading.Lock()
        self.prog_bar_len = 80

    def __call__(self, bytes_amount):
        ''' Call
        When called, increments seen_so_far by bytes_amount,
        calculates percentage of seen_so_far/total file size
        and prints progress bar.
        '''
        # To simplify we'll assume this is hooked up to a single filename.
        with self._lock:
            self._seen_so_far += bytes_amount
            ratio = round((float(self._seen_so_far) / float(self._size)) * (self.prog_bar_len - 6), 1)
            current_length = int(round(ratio))

            percentage = round(100 * ratio / (self.prog_bar_len - 6), 1)

            bars = '+' * current_length
            output = bars + ' ' * (self.prog_bar_len - current_length - len(str(percentage)) - 1) + str(percentage) + '%'

            if self._seen_so_far != self._size:
                sys.stdout.write(output + '\\r')
            else:
                sys.stdout.write(output + '\\n')
            sys.stdout.flush()

def str_to_bool(s):
    if s is None:
      return False
    return s.lower() in ['true', '1', 't', 'y', 'yes']

# TODO: support signature version?
def build_boto3_client():
  return boto3.client(
    's3',
    aws_access_key_id=os.getenv('SDG_OBJECT_STORE_ACCESS_KEY'),
    aws_secret_access_key=os.getenv('SDG_OBJECT_STORE_SECRET_KEY'),
    endpoint_url=os.getenv('SDG_OBJECT_STORE_ENDPOINT', None),
    region_name=os.getenv('SDG_OBJECT_STORE_REGION', None),
    verify=str_to_bool(os.getenv('SDG_OBJECT_STORE_VERIFY_TLS', 'true'))
)

def download_s3_file():
    s3 = build_boto3_client()

    bucket_name = os.getenv('SDG_OBJECT_STORE_BUCKET')
    s3_key = os.getenv('SDG_OBJECT_STORE_DATA_KEY')
    output_file = '{data_pvc_mount_path}/data.tar.gz'

    progress = ProgressPercentage(s3, bucket_name, s3_key)
    s3.download_file(bucket_name, s3_key, output_file, Callback=progress)

def upload_s3_file():
    s3 = build_boto3_client()

    bucket_name = os.getenv('SDG_OBJECT_STORE_BUCKET')
    s3_key = os.getenv('FINAL_DATA_TAR_FILE')
    input_file = os.getenv('FINAL_DATA_TAR_PATH')

    progress = ProgressPercentage(s3, bucket_name, input_file)
    s3.upload_file(input_file, bucket_name, s3_key, Callback=progress)

if __name__ == "__main__":
    if os.getenv('STRATEGY') == 'download':
      print('Downloading file from object store')
      download_s3_file()
    elif os.getenv('STRATEGY') == 'upload':
      print('Uploading file to object store')
      upload_s3_file()
    else:
      raise ValueError('Unknown STRATEGY')
EOF

python "$tmp"/s3.py

if [ "$STRATEGY" == "download" ]; then
    for dir in data model taxonomy; do
        dir_path="{data_pvc_mount_path}/$dir"
        if [ -d "$dir_path" ]; then
            echo "Directory $dir_path exists, it will be overwritten by the content of the archive"
        fi
    done

    echo "Extracting data from the archive"

    tar \
      --touch \
      --no-same-owner \
      --no-same-permissions \
      --directory {data_pvc_mount_path} \
      --extract \
      --verbose \
      --file {data_pvc_mount_path}/data.tar.gz

    # Enable globstar for recursive globbing
    shopt -s globstar

    # Patterns to match
    patterns=(
        "{data_pvc_mount_path}/model/config.json"
        "{data_pvc_mount_path}/model/tokenizer.json"
        "{data_pvc_mount_path}/model/tokenizer_config.json"
        "{data_pvc_mount_path}/model/*.safetensors"
        "{data_pvc_mount_path}/data/skills_recipe_*.yaml"
        "{data_pvc_mount_path}/data/knowledge_recipe_*.yaml"
        "{data_pvc_mount_path}/data/skills_train_*.jsonl"
        "{data_pvc_mount_path}/data/knowledge_train_*.jsonl"
        "{data_pvc_mount_path}/taxonomy/knowledge"
        "{data_pvc_mount_path}/taxonomy/foundational_skills"
    )

    match_count=0
{% raw %}
    for pattern in "${{patterns[@]}}"; do
        matching_files=($pattern)
        if [ ! -s "${{matching_files[0]}}" ]; then
            echo "No files found matching pattern: $pattern: ${{matching_files[0]}}"
        else
            echo "Files found matching pattern: $pattern: ${{matching_files[0]}}"
            match_count=$((match_count+1))
        fi
    done

    if [ $match_count -ne ${{#patterns[@]}} ]; then
        echo "Error: Not all files were found, only $match_count files were found"
        ls -laR {data_pvc_mount_path}
        exit 1
    fi
{% endraw %}

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
        "If used "
        "endpoint, bucket, access_key, secret_key, region, data_key, verify_tls will be ignored."
        "All supported options are: "
        "endpoint, bucket, access_key, secret_key, region, data_key, verify_tls"
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
    "--judge-serving-model-endpoint",
    type=str,
    help=(
        "Judge model serving endpoint for evaluation."
        "e.g. http://serving.kubeflow.svc.cluster.local:8080/v1"
    ),
)
@click.option(
    "--judge-serving-model-name",
    type=str,
    help="The name of the model to use for evaluation.",
)
@click.option(
    "--judge-serving-model-api-key",
    type=str,
    help=(
        "Serving model API key for evaluation. " "(JUDGE_SERVING_MODEL_API_KEY env var)"
    ),
    envvar="JUDGE_SERVING_MODEL_API_KEY",
)
@click.option(
    "--judge-serving-model-secret",
    type=str,
    envvar="JUDGE_SERVING_MODEL_SECRET",
    help=(
        "Name of the Kubernetes Secret containing the judge serving model endpoint. "
        "For evaluation only. "
        "The namespace is inferred from the namespace option. "
        "The following keys are expected: JUDGE_API_KEY, JUDGE_ENDPOINT, JUDGE_NAME "
        " (JUDGE_SERVING_MODEL_SECRET env var)"
        "If used, the --judge-serving-model-{api-key,endpoint,name} options will be ignored."
    ),
)
@click.option(
    "--nproc-per-node",
    type=int,
    help="Number of GPU to use per node - for training only",
    default=1,
)
@click.option(
    "--eval-type",
    help="Type of evaluation to run",
    type=click.Choice([EVAL_TYPE_MT_BENCH, EVAL_TYPE_FINAL]),
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
        "Useful when calling training phases independently "
        "and users wants to point to the epoch directory. "
        "Very advanced usage, not recommended for general use."
    ),
    type=str,
)
@click.option(
    "--sdg-object-store-endpoint",
    envvar="SDG_OBJECT_STORE_ENDPOINT",
    help=(
        "Object store endpoint if different than the official AWS S3 endpoint. "
        "Expects an URL. TLS with self-signed certificates is not supported. "
        "(SDG_OBJECT_STORE_ENDPOINT env var)"
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
        "Name of tarball that contains SDG data AND model files."
        "(SDG_OBJECT_STORE_DATA_KEY env var)."
        "The tarball MUST contain two directories: data and model."
        "The data directory contains the SDG data."
        "The model directory contains the model to train."
        "To archive use the following command: "
        "tar -czvf data.tar.gz /path/to/data /path/to/model /path/to/taxonomy."
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
        "If used "
        "endpoint, bucket, access_key, secret_key, region, data_key, verify_tls will be ignored."
        "All supported options are: "
        "endpoint, bucket, access_key, secret_key, region, data_key, verify_tls"
    ),
    type=str,
)
@click.option(
    "--force-pull",
    help=(
        "Force pull the data (sdg data and model) from the object store "
        "even if it already exists in the PVC."
    ),
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
@click.option(
    "--num-instructions-to-generate",
    help="Number of instructions to generate.",
    default=30,
    hidden=True,
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
    judge_serving_model_endpoint: typing.Optional[str] = None,
    judge_serving_model_name: typing.Optional[str] = None,
    judge_serving_model_api_key: typing.Optional[str] = None,
    judge_serving_model_secret: typing.Optional[str] = None,
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
    num_instructions_to_generate: typing.Optional[int] = 30,
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
        judge_serving_model_endpoint (str): The serving endpoint for evaluation. For Evaluation
        only.
        judge_serving_model_name (str): The serving model name for evaluation. For Evaluation only.
        judge_serving_model_api_key (str): The serving model API key for evaluation. For Evaluation
        only.
        judge_serving_model_secret (str): The name of the Kubernetes Secret containing the serving
        model credentials. For Evaluation only.
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
        sdg_object_store_secret (str): The name of the Kubernetes Secret containing the SDG object
        store credentials. The namespace is inferred from the namespace option.
        force_pull (bool): Force pull the data (sdg data and model) from the object store even if it
        already exists in the PVC.
        training_1_epoch_num (int): Number of epochs to train the model for during phase 1.
        training_2_epoch_num (int): Number of epochs to train the model for during phase 2.
        num_instructions_to_generate (int): Number of instructions to generate during SDG.

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
    ctx.obj["judge_serving_model_endpoint"] = judge_serving_model_endpoint
    ctx.obj["judge_serving_model_name"] = judge_serving_model_name
    ctx.obj["judge_serving_model_api_key"] = judge_serving_model_api_key
    ctx.obj["judge_serving_model_secret"] = judge_serving_model_secret
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
    ctx.obj["num_instructions_to_generate"] = num_instructions_to_generate

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
        ctx.obj["eval_type"] = EVAL_TYPE_MT_BENCH
        scores = ctx.invoke(evaluation)
        scores = json.loads(scores)
        logger.info("Best model: %s", scores.get("best_model"))
        ctx.obj["candidate_model"] = scores.get("best_model")

        # Final evaluation
        ctx.obj["eval_type"] = EVAL_TYPE_FINAL
        ctx.invoke(evaluation)
        logger.info("InstructLab Training Finished!")

        # Push the best model to S3
        ctx.invoke(upload_trained_model)


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
    num_instructions_to_generate: int,
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
        num_instructions_to_generate (int): The number of instructions to generate.
        exec_git_clone_op_repo_url (str): The URL of the taxonomy repository.
        exec_git_clone_op_repo_branch (str, optional): The branch of the taxonomy repository.
        exec_git_clone_op_repo_pr (str, optional): The pull request number of the taxonomy
        repository.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """
    # Configureate Pod template container
    exec_sdg_op_command = """
{{exec_sdg_op_command}}
"""
    exec_sdg_op_args = f"""
{{exec_sdg_op_args}}
"""
    exec_huggingface_importer_op_command = """
{{exec_huggingface_importer_op_command}}
"""
    exec_huggingface_importer_op_args = f"""
{{exec_huggingface_importer_op_args}}
"""
    exec_data_processing_op_command = """
{{exec_data_processing_op_command}}
"""
    exec_data_processing_op_args = f"""
{{exec_data_processing_op_args}}
"""

    init_containers = [
        kubernetes.client.V1Container(
            name="sdg-op-fetch-taxonomy-data",
            image=DS_IMAGE,
            command=["/bin/sh", "-c"],
            args={{exec_git_clone_op_args}},
            volume_mounts=get_vol_mount(),
            security_context=get_security_context(),
        ),
        kubernetes.client.V1Container(
            name="sdg-op-generate-synthetic-data",
            image=RHELAI_IMAGE,
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
            image=RHELAI_IMAGE,
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
        image=DS_IMAGE,
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


def create_data_job(
    namespace: str,
    job_name: str,
    sdg_object_store_secret: str,
    strategy: str,
    force_pull: bool = False,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to fetch SDG data from an object
    store.

    Args:
        namespace (str): The namespace in which the job will be created.
        job_name (str): The name of the job.
        sdg_object_store_secret (str): The name of the Kubernetes Secret containing the SDG object
        store credentials.
        strategy (str): The strategy to use to fetch the data. Either "download" or "upload".
        force_pull (bool): Force pull the data from the object store even if it already exists in
        the PVC.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """

    exec_data_processing_op_command = """
{{exec_data_processing_op_command}}
"""
    exec_data_processing_op_args = f"""
data_processing_op(max_seq_len={MAX_SEQ_LEN}, max_batch_len={MAX_BATCH_LEN}, sdg={DATA_PVC_SDG_PATH}, model={DATA_PVC_SDG_PATH}, processed_data={PREPROCESSED_DATA_PATH})
"""

    data_container = kubernetes.client.V1Container(
        name=f"{strategy}-data-object-store",
        image=DS_IMAGE,
        command=["/bin/sh", "-c"],
        args=[
            DATA_SCRIPT.format(
                strategy=strategy,
                force_pull=force_pull,
                data_pvc_mount_path=DATA_PVC_MOUNT_PATH,
                mt_bench_output_path=MT_BENCH_OUTPUT_PATH,
                mt_bench_scores_path=MT_BENCH_SCORES_PATH,
                mt_bench_branch_scores_path=MT_BENCH_BRANCH_SCORES_PATH,
                mmlu_branch_scores_path=MMLU_BRANCH_SCORES_PATH,
                candidate_model_path=CANDIDATE_MODEL_PATH,
            )
        ],
        volume_mounts=get_vol_mount(),
        env=[
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_ENDPOINT",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="endpoint",
                        optional=True,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_BUCKET",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="bucket",
                        optional=False,
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
                        name=sdg_object_store_secret,
                        key="region",
                        optional=True,
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_DATA_KEY",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret,
                        key="data_key",
                        optional=False,
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

    sdg_data_preprocess_container = kubernetes.client.V1Container(
        name="sdg-data-preprocess",
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

    main_container = None
    if strategy == "download":
        main_container = sdg_data_preprocess_container
    # For the upload strategy, the main container is the data container since we only upload the
    # trained model back to the object store
    elif strategy == "upload":
        main_container = data_container

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "data-" + strategy}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            containers=[main_container],
            volumes=get_vol(),
        ),
    )

    if strategy == "download":
        template.spec.init_containers = [data_container]

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
    eval_type: str,
    nproc_per_node: int = 1,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to run Evaluation steps.

    Args:
        namespace (str): The namespace in which the job will be created.
        eval_type (str): The type of evaluation to run.
        nproc_per_node (int): The number of processes per node.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """

    job_name = f"eval-{eval_type}"

    # if eval_type == "mmlu":
    #     init_containers = [
    #         kubernetes.client.V1Container(
    #             name=f"run-eval-{eval_type}",
    #             image="{{exec_run_mmlu_op_image}}",
    #             command={{exec_run_mmlu_op_command}},
    #             args={{exec_run_mmlu_op_args}},
    #             volume_mounts=[
    #                 kubernetes.client.V1VolumeMount(
    #                     name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
    #                 ),
    #             ],
    #         )
    #     ]
    #     container = kubernetes.client.V1Container(
    #         name=f"output-eval-{eval_type}-scores",
    #         image="{{exec_run_mmlu_op_image}}",
    #         command=["/bin/sh", "-c"],
    #         args=[f"cat {MMLU_SCORES_PATH}"],
    #         volume_mounts=[
    #             kubernetes.client.V1VolumeMount(
    #                 name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
    #             ),
    #         ],
    #     )

    exec_run_mt_bench_op_command = """
{{exec_run_mt_bench_op_command}}
"""
    exec_run_mt_bench_op_args = f"""
{{exec_run_mt_bench_op_args}}
"""
    exec_run_final_eval_op_command = """
{{exec_run_final_eval_op_command}}
"""
    exec_run_final_eval_op_args = f"""
{{exec_run_final_eval_op_args}}
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
    elif eval_type == EVAL_TYPE_FINAL:
        init_containers = [
            kubernetes.client.V1Container(
                name=f"run-eval-{eval_type}",
                image=RHELAI_IMAGE,
                command=["/bin/sh", "-ce"],
                args=[
                    PYTHON_EXECUTOR.format(
                        python_code=exec_run_final_eval_op_command,
                        python_main=exec_run_final_eval_op_args.strip(),
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


def log_pod_containers(
    pod: kubernetes.client.V1Pod, container_type: str, namespace: str
):
    """
    Logs the output of containers in a given pod.

    Args:
        pod (kubernetes.client.V1Pod): The pod object containing the containers.
        container_type (str): The type of containers to log (e.g., 'containers', 'init_containers').
        namespace (str): The namespace in which the pod is located.

    Returns:
        None

    Logs:
        Logs the output of each container in the specified pod to the logger. If the container logs
        cannot be retrieved

    Raises:
        kubernetes.client.rest.ApiException: If there is an error other than a 400 status error when
        retrieving the logs. due to a 400 status error, it continues to the next container.
    """
    core_v1 = kubernetes.client.CoreV1Api()
    containers = getattr(pod.spec, container_type)
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

    # It seems that the watcher suffers from a bug where it misses job events
    # https://github.com/kubernetes-client/python/issues/2238
    # Or connections are dropped
    # https://github.com/kubernetes-client/python/issues/2238
    # Once the library supports Informer API, we can switch to it
    # https://github.com/kubernetes-client/python/issues/868
    # Wait for the job to complete
    w = kubernetes.watch.Watch()
    pod_log = None
    exit_flag = False
    while not exit_flag:  # Keep the watch active
        try:
            for event in w.stream(
                batch_v1.list_namespaced_job,
                namespace=namespace,
                timeout_seconds=60,  # Timeout after 1 minutes
            ):
                job_event = event["object"]
                if job_event.metadata.name != job.metadata.name:
                    continue

                logger.info("Job: %s - %s", job.metadata.name, job_event.status)

                # Handle job completion (successful or failed)
                if job_event.status.succeeded == 1:
                    logger.info("Job completed successfully.")
                    pods = core_v1.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"app={job.spec.template.metadata.labels['app']}",
                    )
                    if pods.items:
                        pod_log = core_v1.read_namespaced_pod_log(
                            name=pods.items[0].metadata.name, namespace=namespace
                        )
                    else:
                        logger.error(
                            "No pods found for job %s. The job exists, but the pods are missing.",
                            job.metadata.name,
                        )
                        pod_log = None
                    w.stop()
                    exit_flag = True  # Set the flag to exit the outer loop
                    break

                elif job_event.status.failed == 1:
                    logger.error("Job failed. Pod logs:")
                    pods = core_v1.list_namespaced_pod(
                        namespace=namespace,
                        label_selector=f"app={job.spec.template.metadata.labels['app']}",
                    )
                    for pod in pods.items:
                        log_pod_containers(pod, "init_containers", namespace)
                        log_pod_containers(pod, "containers", namespace)
                    w.stop()
                    raise RuntimeError("Job failed.")

                else:
                    logger.info(
                        "Job '%s' is still running. Waiting for the next event.",
                        job.metadata.name,
                    )

        except kubernetes.client.exceptions.ApiException as e:
            logger.error("API exception occurred: %s", str(e))
            time.sleep(5)  # Backoff before retrying
        except urllib3.exceptions.ProtocolError as e:
            logger.warning("Connection broken reconnecting the watcher: %s", str(e))
            time.sleep(5)  # Backoff before retrying

        finally:
            w.stop()  # Ensure the watch is stopped after each try

    # Ensure pod logs are returned after success
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
    num_instructions_to_generate = ctx.obj["num_instructions_to_generate"]

    # check in the context
    if not taxonomy_repo_branch and not taxonomy_repo_pr:
        raise ValueError(
            "Either '--taxonomy-repo-branch' or '--taxonomy-repo-pr' "
            "must be provided to the 'run' command."
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
            "access_modes": ["ReadWriteMany"],
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
        num_instructions_to_generate=num_instructions_to_generate,
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
    judge_serving_model_endpoint = ctx.obj["judge_serving_model_endpoint"]
    judge_serving_model_name = ctx.obj["judge_serving_model_name"]
    judge_serving_model_api_key = ctx.obj["judge_serving_model_api_key"]
    judge_serving_model_secret = ctx.obj["judge_serving_model_secret"]
    sdg_object_store_endpoint = ctx.obj["sdg_object_store_endpoint"]
    sdg_object_store_bucket = ctx.obj["sdg_object_store_bucket"]
    sdg_object_store_access_key = ctx.obj["sdg_object_store_access_key"]
    sdg_object_store_secret_key = ctx.obj["sdg_object_store_secret_key"]
    sdg_object_store_region = ctx.obj["sdg_object_store_region"]
    sdg_object_store_data_key = ctx.obj["sdg_object_store_data_key"]
    sdg_object_store_verify_tls = ctx.obj["sdg_object_store_verify_tls"]
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]
    force_pull = ctx.obj["force_pull"]

    # Check if all required arguments are provided for Data Fetch
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
                "'--sdg-object-store-access-key', '--sdg-object-store-secret-key', "
                "'--sdg-object-store-data-key' "
                "must be provided to the 'sdg-data-fetch' command. Alternatively, provide "
                "'--sdg-object-store-secret' to use a Kubernetes Secret."
            )

    # Check if all required arguments are provided for Evaluation
    if not judge_serving_model_secret:
        if not all(
            [
                judge_serving_model_endpoint,
                judge_serving_model_name,
                judge_serving_model_api_key,
            ]
        ):
            # Endpoint is optional if AWS S3 is used
            raise ValueError(
                "All of '--judge-serving-model-endpoint', "
                "'--sdg-object-store-access-key', '--judge-serving-model-name', "
                "'--judge-serving-model-api-key' "
                "must be provided to the 'sdg-data-fetch' command. Alternatively, provide "
                "'--judge-serving-model-secret' to use a Kubernetes Secret."
            )

    logger.info("Running setup for SDG data fetch.")

    # Request the Kubernetes API
    v1 = kubernetes.client.CoreV1Api()

    # SDG Data Fetch secret
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

        if sdg_object_store_verify_tls:
            secret.string_data["verify_tls"] = "true"
        else:
            secret.string_data["verify_tls"] = "false"

        try:
            v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret option is used, verify the presence of the keys and the existence of the secret
    elif sdg_object_store_secret:
        try:
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
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                raise ValueError(
                    f"Secret {sdg_object_store_secret} not found in namespace {namespace}."
                ) from exc

    # Judge serving model secret
    if (
        all(
            [
                judge_serving_model_endpoint,
                judge_serving_model_name,
                judge_serving_model_api_key,
            ]
        )
        and not judge_serving_model_secret
    ):
        judge_serving_model_secret = JUDGE_SERVING_NAME
        secret = kubernetes.client.V1Secret(
            metadata=kubernetes.client.V1ObjectMeta(
                name=judge_serving_model_secret, namespace=namespace
            ),
            string_data={
                "JUDGE_API_KEY": judge_serving_model_api_key,
                "JUDGE_ENDPOINT": judge_serving_model_endpoint,
                "JUDGE_NAME": judge_serving_model_name,
            },
        )

        try:
            v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret option is used, verify the presence of the keys and the existence of the secret
    elif judge_serving_model_secret:
        try:
            secret = v1.read_namespaced_secret(
                name=judge_serving_model_secret, namespace=namespace
            )

            if not all(
                [
                    secret.data.get("JUDGE_API_KEY"),
                    secret.data.get("JUDGE_ENDPOINT"),
                    secret.data.get("JUDGE_NAME"),
                ]
            ):
                raise ValueError(
                    f"The provided secret {judge_serving_model_secret} must contain the keys:"
                    "'JUDGE_API_KEY', 'JUDGE_ENDPOINT', 'JUDGE_NAME' mind the uppercase.",
                )

            judge_serving_model_endpoint = decode_base64(
                secret.data.get("JUDGE_ENDPOINT")
            )
            validate_url(judge_serving_model_endpoint)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                raise ValueError(
                    f"Secret {judge_serving_model_secret} not found in namespace {namespace}."
                ) from exc

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
            logger.info("Successfully created PVC '%s' created.", pvc.get("name"))
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("PVC '%s' already exists.", pvc["name"])
            else:
                raise

    # Create the job to run the pod to execute the SDG data fetch
    job = create_data_job(
        namespace=namespace,
        job_name="data-download",
        sdg_object_store_secret=sdg_object_store_secret,
        strategy="download",
        force_pull=force_pull,
    )

    # Run the job
    run_job(namespace, job)
    logger.info(
        "SDG data, model to train and taxonomy tree were successfully downloaded."
    )


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
            data_pvc_model_path=DATA_PVC_MODEL_PATH,
            data_pvc_sdg_path=DATA_PVC_SDG_PATH,
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
    core_v1 = kubernetes.client.CoreV1Api()
    w = kubernetes.watch.Watch()
    exit_flag = False
    # TODO: this block is getting really deep, would be nice to refactor one day
    while not exit_flag:  # Keep the watch active
        try:
            for event in w.stream(
                api.list_namespaced_custom_object,
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural="pytorchjobs",
            ):
                pytorchjob_event = event["object"]
                if (
                    pytorchjob_event["metadata"]["name"]
                    != pytorch_training_job_yaml["metadata"]["name"]
                ):
                    continue
                pytorchjob_name = pytorchjob_event["metadata"]["name"]

                if (
                    "status" not in pytorchjob_event
                    or "conditions" not in pytorchjob_event["status"]
                ):
                    continue
                logger.info(
                    "PytorchJob: %s - %s",
                    pytorchjob_name,
                    pytorchjob_event["status"].get("conditions", "No conditions yet"),
                )

                master_pod_success = False
                workers_pod_success = {}
                # Always start by the last condition so that if the job is completed, we can stop
                # watching If we don't do this, we might get 'stuck' into the Running condition and
                # never stop
                # watching
                for job_condition in reversed(pytorchjob_event["status"]["conditions"]):
                    print(job_condition)
                    if job_condition["type"] == "Running":
                        # now watch for pod event
                        for event in w.stream(
                            core_v1.list_namespaced_pod,
                            namespace=namespace,
                            label_selector=(
                                f"training.kubeflow.org/job-name=train-phase-{training_phase}"
                            ),
                        ):
                            pod_event = event["object"]
                            if pod_event.metadata.name.startswith(pytorchjob_name):
                                logger.info(
                                    "Pod: %s - %s",
                                    pod_event.metadata.name,
                                    pod_event.status.phase,
                                )
                                for (
                                    container_status
                                ) in pod_event.status.container_statuses:
                                    # We fail on CrashLoopBackOff and not on Error, allowing for
                                    # retries
                                    if (
                                        container_status.state.waiting
                                        and container_status.state.waiting.reason
                                        == "CrashLoopBackOff"
                                    ):
                                        log_pod_containers(
                                            pod_event, "init_containers", namespace
                                        )
                                        log_pod_containers(
                                            pod_event, "containers", namespace
                                        )
                                        raise RuntimeError(
                                            f"Pod {pod_event.metadata.name} failed."
                                        )
                                if pod_event.status.phase == "Failed":
                                    log_pod_containers(
                                        pod_event, "init_containers", namespace
                                    )
                                    log_pod_containers(
                                        pod_event, "containers", namespace
                                    )
                                    w.stop()
                                if pod_event.status.phase == "Succeeded":
                                    if pod_event.metadata.name.startswith(
                                        f"{pytorchjob_name}-master"
                                    ):
                                        master_pod_success = True
                                        logger.info(
                                            "Pod '%s' completed successfully",
                                            pod_event.metadata.name,
                                        )
                                    elif pod_event.metadata.name.startswith(
                                        f"{pytorchjob_name}-worker"
                                    ):
                                        logger.info(
                                            "Pod '%s' completed successfully",
                                            pod_event.metadata.name,
                                        )
                                        # Add the worker pod to the list of successful pods
                                        workers_pod_success[pod_event.metadata.name] = (
                                            True
                                        )
                                    if master_pod_success and (
                                        len(workers_pod_success) == worker_replicas
                                    ):
                                        logger.info(
                                            "All PytorchJob Pods completed successfully"
                                        )
                                        w.stop()
                                        exit_flag = True
                                        # Break here to avoid going into other conditions, we are
                                        # done
                                        break
                                    continue
                    elif job_condition["type"] == "Succeeded":
                        logger.info(
                            "PytorchJob '%s' completed successfully: %s",
                            pytorchjob_name,
                            job_condition["reason"],
                        )
                        logger.info("Training phase %s completed.", training_phase)
                        w.stop()
                        exit_flag = True
                        # Break here to avoid going into other conditions, we are done
                        break
                    elif job_condition["type"] == "Failed":
                        logger.error(
                            "PytorchJob' %s' failed: %s",
                            pytorchjob_name,
                            job_condition["reason"],
                        )
                        w.stop()
                        raise RuntimeError("Job failed.")
        except kubernetes.client.exceptions.ApiException as e:
            logger.error("API exception occurred: %s", str(e))
            time.sleep(5)  # Backoff before retrying
        # Catches the following error:
        # urllib3.exceptions.ProtocolError: ("Connection broken: InvalidChunkLength
        except urllib3.exceptions.ProtocolError as e:
            logger.warning("Connection broken reconnecting the watcher %s", str(e))
            time.sleep(5)  # Backoff before retrying

        finally:
            w.stop()


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
        raise ValueError(
            "Evaluation type must be provided with --eval-type=[mt-bench|final]"
        )

    logger.info("Running %s evaluation.", eval_type)

    # Create and run the evaluation job
    job = create_eval_job(namespace=namespace, eval_type=eval_type)
    scores = run_job(namespace, job)

    if eval_type == EVAL_TYPE_MT_BENCH:
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

        return scores

    logger.info("Evaluation scores: %s", scores)


@run.command(name="upload-trained-model")
@click.pass_context
def upload_trained_model(ctx: click.Context):
    """
    Uploads the trained model back to the object store.

    This function retrieves the namespace and SDG object store secret from the
    provided Click context object. It then creates and runs a data job to
    upload the trained model to the object store.

    Args:
        ctx (click.Context): The Click context object containing command-line
                             parameters and options.

    Returns:
        None

    Raises:
        ValueError: If the SDG object store secret is not provided.
    """
    namespace = ctx.obj["namespace"]
    # At this stage the secret is present from previous phases so no need to check
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]

    logger.info("Uploading the trained model back to the object store.")
    job = create_data_job(
        namespace=namespace,
        job_name="trained-model-upload",
        sdg_object_store_secret=sdg_object_store_secret,
        strategy="upload",
    )

    # Run the job
    run_job(namespace, job)
    logger.info("Successfully uploaded newly trained model back to the object store.")


if __name__ == "__main__":
    # Configs can be set in Configuration class directly or using helper utility
    try:
        kubernetes.config.load_kube_config()
    except kubernetes.config.ConfigException:
        logger.info("Failed to load kube config. Trying in-cluster config")
        kubernetes.config.load_incluster_config()

    cli()
