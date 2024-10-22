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
    - Make sure resources get cleaned up after the job is done. (configmap, secret etc) using a
      finalizer.
    - See if we can use KServe to deploy the model and serve it for SDG Data Generation.
      kubernetes_yaml/mixtral_serve/mixtral_serve.yaml
"""

import base64
import json
import logging
import os
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
SDG_OBJECT_STORE_SECRET_NAME = "sdg-object-store-credentials"
SDG_SERVING_NAME = "sdg-serving-details"
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
PREPROCESSED_DATA_SKILLS_PATH = path.join(PREPROCESSED_DATA_PATH, "skills")
PREPROCESSED_DATA_KNOWLEDGE_PATH = path.join(PREPROCESSED_DATA_PATH, "knowledge")
MT_BENCH_OUTPUT_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-results.txt")
MT_BENCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-best.txt")
MT_BENCH_BRANCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mt-bench-branch-best.txt")
MMLU_BRANCH_SCORES_PATH = path.join(DATA_PVC_MOUNT_PATH, "mmlu-branch-best.txt")
CANDIDATE_MODEL_PATH_PREFIX = path.join(
    DATA_PVC_MOUNT_PATH, "model/output/phase_2/hf_format"
)
CANDIDATE_MODEL_PATH = path.join(CANDIDATE_MODEL_PATH_PREFIX, "candidate_model")
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
JUDGE_CA_CERT_ENV_VAR_NAME = "JUDGE_CA_CERT_PATH"
JUDGE_CA_CERT_PATH = "/tmp"
JUDGE_CA_CERT_CM_KEY = "ca-bundle.crt"

# TEMPLATES
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
                  PATH_TO_DATA={preprocessed_data_knowledge_path}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_DATA="{preprocessed_data_skills_path}"; fi
                  echo "Running phase $phase_num"
                  PATH_TO_MODEL={path_to_model}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/phase_1/hf_format/$(ls --sort=time {path_to_model}/output/phase_1/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  echo "Using $PATH_TO_DATA data for training"
                  mkdir -p {data_pvc_model_path};
                  mkdir -p {data_pvc_sdg_path};
                  mkdir -p {path_to_model}/output/phase_{phase_num}
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path="$PATH_TO_DATA"/data.jsonl \
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
                  PATH_TO_DATA={preprocessed_data_knowledge_path}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_DATA="{preprocessed_data_skills_path}"; fi
                  echo "Running phase $phase_num"
                  PATH_TO_MODEL={path_to_model}
                  if [ "$phase_num" -eq 2 ]; then PATH_TO_MODEL="{path_to_model}/output/phase_1/hf_format/$(ls --sort=time {path_to_model}/output/phase_1/hf_format|head -n 1)"; fi
                  echo "Using $PATH_TO_MODEL model for training"
                  echo "Using $PATH_TO_DATA data for training"
                  tmp_model=$(mktemp -d)
                  mkdir -p "$tmp_model";
                  torchrun --nnodes {nnodes} \
                    --nproc_per_node {nproc_per_node} \
                    --node_rank $(RANK) \
                    --rdzv_endpoint $(MASTER_ADDR):$(MASTER_PORT) \
                    -m instructlab.training.main_ds \
                    --model_name_or_path="$PATH_TO_MODEL" \
                    --data_path="$PATH_TO_DATA"/data.jsonl \
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

export SDG_IN_CLUSTER={sdg_in_cluster}
export STRATEGY={strategy}

if [ -z "$STRATEGY" ] || [ "$STRATEGY" == "None" ]; then
    echo "STRATEGY is not set - must be 'download' or 'upload'"
    exit 1
fi

if [ "$STRATEGY" == "download" ]; then
    FORCE_PULL={force_pull}
    if [ -s {data_pvc_mount_path}/data.tar.gz ]; then
        echo "Data tarball already exists in the PVC. Skipping download and extraction."
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
    if [ "$SDG_IN_CLUSTER" == "True" ]; then
        # When SDG is in-cluster we will run synthetic data generation in the cluster, so the "data"
        directory is not needed since it will be generated
        patterns=(
            "{data_pvc_mount_path}/model/config.json"
            "{data_pvc_mount_path}/model/tokenizer.json"
            "{data_pvc_mount_path}/model/tokenizer_config.json"
            "{data_pvc_mount_path}/model/*.safetensors"
            "{data_pvc_mount_path}/taxonomy/knowledge"
            "{data_pvc_mount_path}/taxonomy/foundational_skills"
        )
    # when not in-cluster it is retrieved data must exist
    else
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
    fi

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
      containers:
      - name: {name}
        image: {image}
        command:
          - "python3"
          - "/config/{script_name}"
          - "run"
        args: {args}
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
    help="Name of the standalone script in the ConfigMap (key)",
    default="standalone",
)
@click.option(
    "--args",
    type=str,
    help="Extra arguments to pass to the script",
    multiple=True,
    required=True,
)
def show(
    namespace: str,
    name: str,
    image: str,
    script_configmap: str,
    script_name: str,
    service_account: str,
    args: typing.List[str],
):
    """
    Print an example Job YAML to stdout to run the script in a Kubernetes cluster.
    The job excepts the standalone.py script to be available in a ConfigMap.
    """
    script = yaml.safe_load(
        JOB_SCRIPT_EXAMPLE.format(
            name=name,
            namespace=namespace,
            image=image,
            script_configmap=script_configmap,
            script_name=script_name,
            args=list(args),
        )
    )

    if service_account:
        script["spec"]["template"]["spec"]["serviceAccountName"] = service_account

    print(yaml.dump(script))


@cli.group(invoke_without_command=True)
@click.option("--namespace", type=str, help="Kubernetes namespace to use")
@click.option(
    "--taxonomy-repo-branch",
    type=str,
    help="Branch of the taxonomy repository - for SDG only",
)
@click.option(
    "--taxonomy-repo-pr",
    type=str,
    help="Pull request number of the taxonomy repository - for SDG only",
)
@click.option(
    "--storage-class",
    type=str,
    help="Storage class to use for the PersistentVolumeClaim - for SDG only",
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
    "--judge-serving-model-ca-cert",
    type=str,
    help=(
        "Name of the Kubernetes ConfigMap containing the serving model CA cert."
        "The expected key name is 'ca-bundle.crt'."
    ),
)
@click.option(
    "--judge-serving-model-ca-cert-cm-key",
    type=str,
    help="Name of the Key in the Kubernetes ConfigMap containing the serving model CA cert.",
    default=JUDGE_CA_CERT_CM_KEY,
)
@click.option(
    "--judge-serving-model-secret",
    type=str,
    envvar="JUDGE_SERVING_MODEL_SECRET",
    help=(
        "Name of the Kubernetes Secret containing the judge serving model endpoint. "
        "For evaluation only. "
        "The namespace is inferred from the namespace option. "
        "The following keys are expected: JUDGE_API_KEY, JUDGE_ENDPOINT, JUDGE_NAME"
        "Optional keys are: JUDGE_CA_CERT, JUDGE_CA_CERT_CM_KEY"
        " (JUDGE_SERVING_MODEL_SECRET env var)"
        "If used, --judge-serving-model-{api-key,endpoint,name,ca-cert} will be ignored."
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
    "--sdg-serving-model-endpoint",
    type=str,
    help=(
        "SDG model serving endpoint."
        "e.g. http://serving.kubeflow.svc.cluster.local:8080/v1"
    ),
)
@click.option(
    "--sdg-serving-model-name",
    type=str,
    help="The name of the model on the serving endpoint.",
)
@click.option(
    "--sdg-serving-model-api-key",
    type=str,
    help="Serving model API key for SDG. (SDG_SERVING_MODEL_API_KEY env var)",
    envvar="SDG_SERVING_MODEL_API_KEY",
)
@click.option(
    "--sdg-serving-model-secret",
    type=str,
    envvar="SDG_SERVING_MODEL_SECRET",
    help=(
        "Name of the Kubernetes Secret containing the sdg serving model endpoint. "
        "For SDG only. "
        "The namespace is inferred from the namespace option. "
        "The following keys are expected: endpoint, model, api_key "
        " (SDG_SERVING_MODEL_SECRET env var)"
        "If used, the --sdg-serving-model-{api-key,endpoint,name} options will be ignored."
    ),
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
    help="Number of instructions to generate. For SDG only.",
    default=30,
)
@click.option(
    "--dry-run",
    help=(
        "Print the generated YAML to stdout instead of creating the resources."
        "**WARNING**: secrets will be printed too!"
    ),
    is_flag=True,
)
@click.option(
    "--sdg-in-cluster",
    help="Run SDG in the cluster. Default is retrieve SDG Data from object store.",
    default=False,
    is_flag=True,
)
@click.pass_context
def run(
    ctx: click.Context,
    namespace: typing.Optional[str] = None,
    taxonomy_repo_branch: typing.Optional[str] = "",
    taxonomy_repo_pr: typing.Optional[str] = "",
    storage_class: typing.Optional[str] = None,
    judge_serving_model_endpoint: typing.Optional[str] = None,
    judge_serving_model_name: typing.Optional[str] = None,
    judge_serving_model_api_key: typing.Optional[str] = None,
    judge_serving_model_ca_cert: typing.Optional[str] = None,
    judge_serving_model_ca_cert_cm_key: typing.Optional[str] = None,
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
    sdg_serving_model_secret: typing.Optional[str] = None,
    sdg_serving_model_endpoint: typing.Optional[str] = None,
    sdg_serving_model_name: typing.Optional[str] = None,
    sdg_serving_model_api_key: typing.Optional[str] = None,
    force_pull: typing.Optional[bool] = False,
    training_1_epoch_num: int = 7,
    training_2_epoch_num: int = 10,
    num_instructions_to_generate: typing.Optional[int] = 30,
    dry_run: bool = False,
    sdg_in_cluster: bool = False,
):
    """
    Execute the distributed training on Kubernetes.

    Args:
        namespace (str): The namespace to use for the setup process.
        taxonomy_repo_branch (str): The branch of the taxonomy repository. For SDG only.
        taxonomy_repo_pr (int): The pull request number of the taxonomy repository. For SDG only.
        storage_class (str): The storage class to use for the PersistentVolumeClaim. For SDG only.
        judge_serving_model_endpoint (str): The serving endpoint for evaluation. For Evaluation
        only.
        judge_serving_model_name (str): The serving model name for evaluation. For Evaluation only.
        judge_serving_model_api_key (str): The serving model API key for evaluation. For Evaluation
        only.
        judge_serving_model_ca_cert (str): The serving model CA cert for evaluation.
        judge_serving_model_ca_cert_cm_key (str): The name of the Key in the Kubernetes ConfigMap
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
        sdg_serving_model_secret (str): The name of the Kubernetes Secret containing the model
        serving details. For SDG only.
        sdg_serving_model_endpoint (str): The serving endpoint for SDG.
        sdg_serving_model_name (str): The serving model name for SDG.
        sdg_serving_model_api_key (str): The serving model API key for SDG.
        force_pull (bool): Force pull the data (sdg data and model) from the object store even if it
        already exists in the PVC.
        training_1_epoch_num (int): Number of epochs to train the model for during phase 1.
        training_2_epoch_num (int): Number of epochs to train the model for during phase 2.
        num_instructions_to_generate (int): Number of instructions to generate during SDG.
        dry_run (bool): Print the generated YAML to stdout instead of creating the resources.
        sdg_in_cluster (bool): Run SDG in the cluster. Default is retrieve SDG Data from an object store.
    Returns:
        None
    """
    ctx.ensure_object(dict)
    ctx.obj["namespace"] = namespace
    ctx.obj["taxonomy_repo_branch"] = taxonomy_repo_branch
    ctx.obj["taxonomy_repo_pr"] = taxonomy_repo_pr
    ctx.obj["storage_class"] = storage_class
    ctx.obj["judge_serving_model_endpoint"] = judge_serving_model_endpoint
    ctx.obj["judge_serving_model_name"] = judge_serving_model_name
    ctx.obj["judge_serving_model_api_key"] = judge_serving_model_api_key
    ctx.obj["judge_serving_model_ca_cert"] = judge_serving_model_ca_cert
    ctx.obj["judge_serving_model_secret"] = judge_serving_model_secret
    ctx.obj["judge_serving_model_ca_cert_cm_key"] = judge_serving_model_ca_cert_cm_key
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
    ctx.obj["sdg_serving_model_secret"] = sdg_serving_model_secret
    ctx.obj["sdg_serving_model_endpoint"] = sdg_serving_model_endpoint
    ctx.obj["sdg_serving_model_name"] = sdg_serving_model_name
    ctx.obj["sdg_serving_model_api_key"] = sdg_serving_model_api_key
    ctx.obj["force_pull"] = force_pull
    ctx.obj["training_1_epoch_num"] = training_1_epoch_num
    ctx.obj["training_2_epoch_num"] = training_2_epoch_num
    ctx.obj["num_instructions_to_generate"] = num_instructions_to_generate
    ctx.obj["dry_run"] = dry_run
    ctx.obj["sdg_in_cluster"] = sdg_in_cluster

    ##########################
    # MAIN WORKFLOW SEQUENCE #
    ##########################
    # When the script is simply called like: 'python standalone.py run'
    # We will run the entire workflow
    if ctx.invoked_subcommand is None:
        if sdg_in_cluster:
            ctx.invoke(sdg)
        else:
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
        if not dry_run:
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


def create_sdg_container(
    sdg_serving_model_secret: str,
    num_instructions_to_generate: int = 30,
    exec_git_clone_op_repo_branch: str = "",
    exec_git_clone_op_repo_pr: str = "",
) -> kubernetes.client.V1Container:
    """
    Creates a Kubernetes V1Job container for generating synthetic data.

    This function configures a Pod template container with the specified
    command and arguments for executing synthetic data generation operations.
    It sets up the container with the necessary image, command, arguments,
    volume mounts, and security context.

    Args:
        sdg_object_store_secret (str): The name of the Kubernetes Secret containing the SDG object
        store credentials.

    Returns:
        kubernetes.client.V1Job: A configured Kubernetes V1Job container.
    """
    # Configureate Pod template container
    exec_sdg_op_command = """
{{exec_sdg_op_command}}
"""
    exec_sdg_op_args = f"""
{{exec_sdg_op_args}}
"""

    return kubernetes.client.V1Container(
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
                secret_ref=kubernetes.client.V1SecretEnvSource(
                    name=sdg_serving_model_secret
                )
            ),
        ],
    )


def create_data_job(
    namespace: str,
    job_name: str,
    sdg_object_store_secret: str,
    strategy: str,
    sdg_serving_model_secret: str = None,
    force_pull: bool = False,
    sdg_in_cluster: bool = False,
    num_instructions_to_generate: int = 30,
    taxonomy_repo_pr: str = "",
    taxonomy_repo_branch: str = "",
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
        sdg_serving_model_secret (str): The name of the Kubernetes Secret containing the SDG
        serving endpoint details.
        strategy (str): The strategy to use to fetch the data. Either "download" or "upload".
        force_pull (bool): Force pull the data from the object store even if it already exists in
        the PVC.

    Returns:
        kubernetes.client.V1Job: A Kubernetes Job object configured with the specified parameters.
    """

    labels = {"app": "data-" + strategy}
    if sdg_in_cluster:
        labels["app"] = "sdg"

    exec_data_processing_op_command = """
{{exec_data_processing_op_command}}
"""
    exec_data_processing_op_args = f"""
data_processing_op(max_seq_len={MAX_SEQ_LEN}, max_batch_len={MAX_BATCH_LEN}, sdg="{DATA_PVC_SDG_PATH}", model="{DATA_PVC_MODEL_PATH}", skills_processed_data="{PREPROCESSED_DATA_SKILLS_PATH}", knowledge_processed_data="{PREPROCESSED_DATA_KNOWLEDGE_PATH}")
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
                sdg_in_cluster=sdg_in_cluster,
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
        metadata=kubernetes.client.V1ObjectMeta(labels=labels),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            containers=[main_container],
            volumes=get_vol(),
        ),
    )

    if strategy == "download":
        init_containers = [data_container]
        # If sdg_in_cluster is True, we append the create_sdg_container to the init_containers so
        # the sequence will be: download data (model and taxonomy) -> generate synthetic data
        if sdg_in_cluster:
            init_containers.append(
                create_sdg_container(
                    sdg_serving_model_secret,
                    num_instructions_to_generate=num_instructions_to_generate,
                    exec_git_clone_op_repo_branch=taxonomy_repo_branch,
                    exec_git_clone_op_repo_pr=taxonomy_repo_pr,
                )
            )
        template.spec.init_containers = init_containers

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
    judge_serving_model_secret: str,
    nproc_per_node: int = 1,
    judge_serving_model_ca_cert: str = None,
    judge_serving_model_ca_cert_cm_key: str = None,
) -> kubernetes.client.V1Job:
    """
    Create a Kubernetes Job object.

    This function generates a Kubernetes Job object configured to run Evaluation steps.

    Args:
        namespace (str): The namespace in which the job will be created.
        eval_type (str): The type of evaluation to run.
        judge_serving_model_secret (str): The name of the Kubernetes Secret containing the judge
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

    eval_container = kubernetes.client.V1Container(
        name=f"run-eval-{eval_type}",
        image=RHELAI_IMAGE,
        command=["/bin/sh", "-ce"],
        volume_mounts=get_vol_mount(),
        security_context=get_security_context(),
        env_from=[
            kubernetes.client.V1EnvFromSource(
                secret_ref=kubernetes.client.V1SecretEnvSource(
                    name=judge_serving_model_secret
                )
            ),
        ],
        resources=kubernetes.client.V1ResourceRequirements(
            requests={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
            limits={"cpu": "1", "nvidia.com/gpu": nproc_per_node},
        ),
    )
    eval_args = {
        EVAL_TYPE_MT_BENCH: [
            PYTHON_EXECUTOR.format(
                python_code=exec_run_mt_bench_op_command,
                python_main=exec_run_mt_bench_op_args.strip(),
            ),
        ],
        EVAL_TYPE_FINAL: [
            PYTHON_EXECUTOR.format(
                python_code=exec_run_final_eval_op_command,
                python_main=exec_run_final_eval_op_args.strip(),
            ),
        ],
    }
    try:
        eval_container.args = eval_args[eval_type]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation type: {eval_type}") from exc

    init_containers = [eval_container]

    output_container = kubernetes.client.V1Container(
        name=f"output-eval-{eval_type}-scores",
        image=RHELAI_IMAGE,
        command=["/bin/sh", "-c"],
        security_context=get_security_context(),
        volume_mounts=get_vol_mount(),
    )
    eval_paths = {
        EVAL_TYPE_MT_BENCH: MT_BENCH_SCORES_PATH,
        EVAL_TYPE_FINAL: MT_BENCH_BRANCH_SCORES_PATH,
    }
    try:
        output_container.args = [f"cat {eval_paths[eval_type]}"]
    except KeyError as exc:
        raise ValueError(f"Unknown evaluation type: {eval_type}") from exc

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": f"eval-{eval_type}"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
            init_containers=init_containers,
            containers=[output_container],
            volumes=get_vol(),
        ),
    )

    if judge_serving_model_ca_cert:
        # Define the volume that references the ConfigMap
        cm_volume = kubernetes.client.V1Volume(
            name="judge-ca-cert-volume",
            config_map=kubernetes.client.V1ConfigMapVolumeSource(
                name=judge_serving_model_ca_cert
            ),
        )
        # Define the volume mount to specify where the Secret should be mounted in the container
        cm_volume_mount = kubernetes.client.V1VolumeMount(
            name="judge-ca-cert-volume",
            mount_path=JUDGE_CA_CERT_PATH,  # Path where the Secret will be mounted
        )
        # Add an env var to the container to specify the path to the CA cert
        eval_container.env.append(
            kubernetes.client.V1EnvVar(
                name=JUDGE_CA_CERT_ENV_VAR_NAME,
                value=os.path.join(
                    JUDGE_CA_CERT_PATH, judge_serving_model_ca_cert_cm_key
                ),
            )
        )
        # Add the volume to the Pod spec
        eval_container.volume_mounts.append(cm_volume_mount)
        # Add the volume mount to the container
        template.spec.volumes.append(cm_volume)

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


def initial_setup(ctx: click.Context) -> None:
    """
    Perform the initial setup for SDG data fetch and judge serving model.

    This function initializes the necessary Kubernetes secrets and Persistent Volume Claims (PVCs)
    based on the provided context. It validates the required parameters and either creates or
    verifies

    Args:
        ctx (click.Context): The Click context object containing the necessary parameters.

    Raises:
        ValueError: If required parameters are missing or if the provided secrets do not contain the
                    necessary keys.
        kubernetes.client.rest.ApiException: If there is an error interacting with the Kubernetes
        API.
    """
    # Populate variables from context
    namespace = ctx.obj["namespace"]
    storage_class = ctx.obj["storage_class"]
    judge_serving_model_endpoint = ctx.obj["judge_serving_model_endpoint"]
    judge_serving_model_name = ctx.obj["judge_serving_model_name"]
    judge_serving_model_api_key = ctx.obj["judge_serving_model_api_key"]
    judge_serving_model_ca_cert = ctx.obj["judge_serving_model_ca_cert"]
    judge_serving_model_ca_cert_cm_key = ctx.obj["judge_serving_model_ca_cert_cm_key"]
    judge_serving_model_secret = ctx.obj["judge_serving_model_secret"]
    sdg_object_store_endpoint = ctx.obj["sdg_object_store_endpoint"]
    sdg_object_store_bucket = ctx.obj["sdg_object_store_bucket"]
    sdg_object_store_access_key = ctx.obj["sdg_object_store_access_key"]
    sdg_object_store_secret_key = ctx.obj["sdg_object_store_secret_key"]
    sdg_object_store_region = ctx.obj["sdg_object_store_region"]
    sdg_object_store_data_key = ctx.obj["sdg_object_store_data_key"]
    sdg_object_store_verify_tls = ctx.obj["sdg_object_store_verify_tls"]
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]
    dry_run = ctx.obj["dry_run"]

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
                "'--judge-serving-model-api-key', '--judge-serving-model-name', "
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
            if dry_run:
                logger.info(
                    "Dry run: Secret would be created.\n%s", secret.metadata.name
                )
            else:
                v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret option is used, verify the presence of the keys and the existence of the secret
    elif sdg_object_store_secret:
        if not dry_run:
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
            if dry_run:
                logger.info(
                    "Dry run: Secret would be created.\n%s", secret.metadata.name
                )
                print(secret)
            else:
                v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret option is used, verify the presence of the keys and the existence of the secret
    elif judge_serving_model_secret:
        if not dry_run:
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

                # Validation of the secret's existence is done in the next conditional block
                if secret.data.get("JUDGE_CA_CERT"):
                    judge_serving_model_ca_cert = secret.data.get("JUDGE_CA_CERT")
                if secret.data.get("JUDGE_CA_CERT_CM_KEY"):
                    judge_serving_model_ca_cert_cm_key = secret.data.get(
                        "JUDGE_CA_CERT_CM_KEY"
                    )
            except kubernetes.client.rest.ApiException as exc:
                if exc.status == 404:
                    raise ValueError(
                        f"Secret {judge_serving_model_secret} not found in namespace {namespace}."
                    ) from exc

    # If the CA cert is provided, verify the existence of the secret
    # We don't add the CA Cert Secret name into the Secret that contains the judge details
    # If provided, the Secret will be mounted as a volume in the evaluation job
    if judge_serving_model_ca_cert and not dry_run:
        try:
            cm = v1.read_namespaced_config_map(
                name=judge_serving_model_ca_cert, namespace=namespace
            )
            # Validate the presence of the key
            if not cm.data.get(judge_serving_model_ca_cert_cm_key):
                raise ValueError(
                    f"Provided ConfigMap {judge_serving_model_ca_cert} does not contain the key:"
                    f"'{judge_serving_model_ca_cert_cm_key}'."
                    "Use '--judge-serving-model-ca-cert-cm-key' to specify the key."
                )
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 404:
                raise ValueError(
                    f"ConfigMap {judge_serving_model_ca_cert} not found in namespace {namespace}."
                ) from exc

    # Assign sdg_object_store_secret and judge_serving_model_secret to the context
    ctx.obj["sdg_object_store_secret"] = sdg_object_store_secret
    # Set the judge secret in the context for the evaluation job
    ctx.obj["judge_serving_model_secret"] = judge_serving_model_secret

    # Set the judge CA cert in the context for the evaluation job, this handles the case where the
    # secret is not provided via the cli flag but inside the secret
    ctx.obj["judge_serving_model_ca_cert"] = judge_serving_model_ca_cert
    ctx.obj["judge_serving_model_ca_cert_cm_key"] = judge_serving_model_ca_cert_cm_key

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
            if dry_run:
                logger.info("Dry run: PVC would be created.\n%s", create_pvc(**pvc))
            else:
                v1.create_namespaced_persistent_volume_claim(
                    namespace=namespace, body=create_pvc(**pvc)
                )
                logger.info("Successfully created PVC '%s' created.", pvc.get("name"))
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("PVC '%s' already exists.", pvc["name"])
            else:
                raise


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
    dry_run = ctx.obj["dry_run"]
    force_pull = ctx.obj["force_pull"]
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]
    sdg_serving_model_secret = ctx.obj["sdg_serving_model_secret"]
    sdg_serving_model_endpoint = ctx.obj["sdg_serving_model_endpoint"]
    sdg_serving_model_name = ctx.obj["sdg_serving_model_name"]
    sdg_serving_model_api_key = ctx.obj["sdg_serving_model_api_key"]
    num_instructions_to_generate = ctx.obj["num_instructions_to_generate"]
    taxonomy_repo_pr = ctx.obj["taxonomy_repo_pr"]
    taxonomy_repo_branch = ctx.obj["taxonomy_repo_branch"]

    v1 = kubernetes.client.CoreV1Api()
    # Secret details validation here!
    # Check if all required arguments are provided for Data Fetch
    if not sdg_serving_model_secret:
        if not all(
            [
                sdg_serving_model_endpoint,
                sdg_serving_model_name,
                sdg_serving_model_api_key,
            ]
        ):
            # Endpoint is optional if AWS S3 is used
            raise ValueError(
                "All of '--sdg-serving-model-endpoint', "
                "'--sdg-serving-model-name', '--sdg-serving-model-api-key', "
                "must be provided to the 'sdg' command. Alternatively, provide "
                "'--sdg-serving-model-secret' to use a Kubernetes Secret."
            )

    # SDG secret
    if (
        # Endpoint (if AWS S3 is used) and Region are optional
        all(
            [
                sdg_serving_model_endpoint,
                sdg_serving_model_name,
                sdg_serving_model_api_key,
            ]
        )
        and not sdg_serving_model_secret
    ):
        validate_url(sdg_serving_model_endpoint)
        sdg_serving_model_secret = SDG_SERVING_NAME
        secret = kubernetes.client.V1Secret(
            metadata=kubernetes.client.V1ObjectMeta(
                name=sdg_serving_model_secret, namespace=namespace
            ),
            string_data={
                "api_key": sdg_serving_model_api_key,
                "endpoint": sdg_serving_model_endpoint,
                "model": sdg_serving_model_name,
            },
        )

        try:
            if dry_run:
                logger.info(
                    "Dry run: Secret would be created.\n%s", secret.metadata.name
                )
            else:
                v1.create_namespaced_secret(namespace=namespace, body=secret)
        except kubernetes.client.rest.ApiException as exc:
            if exc.status == 409:
                logger.info("Secret '%s' already exists.", secret.metadata.name)
            else:
                raise

    # If the secret option is used, verify the presence of the keys and the existence of the secret
    elif sdg_serving_model_secret:
        if not dry_run:
            try:
                secret = v1.read_namespaced_secret(
                    name=sdg_serving_model_secret, namespace=namespace
                )

                def decode_base64(data):
                    return base64.b64decode(data).decode("utf-8")

                if not all(
                    [
                        secret.data.get("api_key"),
                        secret.data.get("model"),
                        secret.data.get("endpoint"),
                    ]
                ):
                    raise ValueError(
                        f"The provided secret {sdg_serving_model_secret} must contain the keys:"
                        "'api_key', 'endpoint', 'model'.",
                    )

                # Validate the endpoint
                endpoint = decode_base64(secret.data.get("endpoint"))
                validate_url(endpoint)
            except kubernetes.client.rest.ApiException as exc:
                if exc.status == 404:
                    raise ValueError(
                        f"Secret {sdg_serving_model_secret} not found in namespace {namespace}."
                    ) from exc

    logger.info("Initial configuration.")
    initial_setup(ctx)

    logger.info("Running setup for SDG.")

    # Create the job to run the pod to execute the SDG data fetch
    job = create_data_job(
        namespace=namespace,
        job_name="sdg",
        sdg_object_store_secret=sdg_object_store_secret,
        sdg_serving_model_secret=sdg_serving_model_secret,
        strategy="download",
        force_pull=force_pull,
        sdg_in_cluster=True,
        num_instructions_to_generate=num_instructions_to_generate,
        taxonomy_repo_pr=taxonomy_repo_pr,
        taxonomy_repo_branch=taxonomy_repo_branch,
    )

    if dry_run:
        logger.info("Dry run: Job would be created.\n%s", job)
        return

    # Run the job
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
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]
    force_pull = ctx.obj["force_pull"]
    dry_run = ctx.obj["dry_run"]

    logger.info("Initial configuration.")
    initial_setup(ctx)

    # Create the job to run the pod to execute the SDG data fetch
    job = create_data_job(
        namespace=namespace,
        job_name="data-download",
        sdg_object_store_secret=sdg_object_store_secret,
        strategy="download",
        force_pull=force_pull,
    )

    if dry_run:
        logger.info("Dry run: Job would be created.\n%s", job)
        return

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
    dry_run = ctx.obj["dry_run"]

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
            preprocessed_data_skills_path=PREPROCESSED_DATA_SKILLS_PATH,
            preprocessed_data_knowledge_path=PREPROCESSED_DATA_KNOWLEDGE_PATH,
        )
    )

    if dry_run:
        logger.info(
            "Dry run: PytorchJob would be created.\n%s", pytorch_training_job_yaml
        )
        return

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
            logger.info("Watching for PytorchJob")
            for event in w.stream(
                api.list_namespaced_custom_object,
                group="kubeflow.org",
                version="v1",
                namespace=namespace,
                plural="pytorchjobs",
                timeout_seconds=60,  # Timeout after 1 minutes
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
                for job_condition in reversed(pytorchjob_event["status"]["conditions"]):
                    if job_condition["type"] == "Running":
                        try:
                            # List PytorchJob Pods
                            pods = core_v1.list_namespaced_pod(
                                namespace=namespace,
                                label_selector=(
                                    f"training.kubeflow.org/job-name=train-phase-{training_phase}"
                                ),
                            )
                            for pod_event in pods.items:
                                if pod_event.metadata.name.startswith(pytorchjob_name):
                                    logger.info(
                                        "Pod: %s - %s",
                                        pod_event.metadata.name,
                                        pod_event.status.phase,
                                    )
                                    # First look if any container is in CrashLoopBackOff
                                    for (
                                        container_status
                                    ) in pod_event.status.container_statuses:
                                        # We fail on CrashLoopBackOff and not on Error, allowing
                                        # for retries
                                        if (
                                            container_status.state.waiting
                                            and container_status.state.waiting.reason
                                            == "CrashLoopBackOff"
                                        ):
                                            log_pod_containers(
                                                pod_event,
                                                "init_containers",
                                                namespace,
                                            )
                                            log_pod_containers(
                                                pod_event, "containers", namespace
                                            )
                                            raise RuntimeError(
                                                f"Pod {pod_event.metadata.name} failed."
                                            )

                                    # If the pod is in a failed state, log the containers and
                                    # stop the watcher
                                    if pod_event.status.phase == "Failed":
                                        log_pod_containers(
                                            pod_event, "init_containers", namespace
                                        )
                                        log_pod_containers(
                                            pod_event, "containers", namespace
                                        )
                                        w.stop()
                                        raise RuntimeError(
                                            f"Pod {pod_event.metadata.name} failed."
                                        )
                        except kubernetes.client.exceptions.ApiException as e:
                            logger.error("API exception occurred: %s", str(e))
                            time.sleep(5)  # Backoff before retrying
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
    dry_run = ctx.obj["dry_run"]
    judge_serving_model_secret = ctx.obj["judge_serving_model_secret"]
    judge_serving_model_ca_cert = ctx.obj["judge_serving_model_ca_cert"]
    judge_serving_model_ca_cert_cm_key = ctx.obj["judge_serving_model_ca_cert_cm_key"]

    # This should only happen if the script is called with the "evaluation" subcommand
    if not judge_serving_model_secret:
        raise ValueError(
            "Judge serving model secret must be provided with --judge-serving-model-secret."
        )

    if eval_type is None:
        raise ValueError(
            "Evaluation type must be provided with --eval-type=[mt-bench|final]"
        )

    logger.info("Running %s evaluation.", eval_type)

    # Create and run the evaluation job
    job = create_eval_job(
        namespace=namespace,
        eval_type=eval_type,
        judge_serving_model_secret=judge_serving_model_secret,
        judge_serving_model_ca_cert=judge_serving_model_ca_cert,
        judge_serving_model_ca_cert_cm_key=judge_serving_model_ca_cert_cm_key,
    )

    if dry_run:
        logger.info("Dry run: Job would be created.\n%s", job)
        return

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
    dry_run = ctx.obj["dry_run"]

    if not sdg_object_store_secret:
        raise ValueError(
            "SDG object store secret must be provided with --sdg-object-store-secret."
        )

    logger.info("Uploading the trained model back to the object store.")
    job = create_data_job(
        namespace=namespace,
        job_name="trained-model-upload",
        sdg_object_store_secret=sdg_object_store_secret,
        strategy="upload",
    )

    if dry_run:
        logger.info("Dry run: Job would be created.\n%s", job)
        return

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
