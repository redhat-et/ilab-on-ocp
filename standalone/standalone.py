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
PYTHON_IMAGE = "registry.access.redhat.com/ubi9/python-311:latest"
SDG_PVC_NAME = "sdg-data"
SDG_PVC_MOUNT_PATH = "/input_data"
SDG_VOLUME_NAME = "input-data"
MODEL_PVC_NAME = "model"
MODEL_PVC_MOUNT_PATH = "/input_model"
MODEL_VOLUME_NAME = "model"
TAXONOMY_PATH = SDG_PVC_MOUNT_PATH + "/taxonomy"
TRAINING_PVC_NAME = "training-data"
TRAINING_PVC_MOUNT_PATH = "/output"
TRAINING_VOLUME_NAME = "output"
PYTORCH_NNODES = 2
PYTORCH_IMAGE = "quay.io/shanand/test-train:0.0.4"
# MMLU_SCORES_PATH = "/output/mmlu-results.txt"
MT_BENCH_SCORES_PATH = "/output/mt-bench-results.txt"
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
                  mkdir -p /output/model;
                  mkdir -p /output/data;
                  python3.11 -u run_main_ds.py --model_path {path_to_model} --ckpt_output_dir /output/model --data_output_dir /input_data/processed_data
              command:
                - /bin/bash
                - '-c'
                - '--'
              image: {PYTORCH_IMAGE}
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
                  value: \"{PYTORCH_NNODES}\"
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
                  mkdir -p /tmp/model;
                  python3.11 -u run_main_ds.py --model_path {path_to_model} --ckpt_output_dir /tmp/model --data_output_dir /input_data/processed_data
              command:
                - /bin/bash
                - '-c'
                - '--'
              image: {PYTORCH_IMAGE}
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
                  value: \"{PYTORCH_NNODES}\"
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
# TODO: support signature version?
SDG_DATA_SCRIPT = """
set -e

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
    output_file = '{SDG_PVC_MOUNT_PATH}/sdg.tar.gz'

    s3.download_file(bucket_name, s3_key, output_file)

def upload_s3_file():
    s3 = build_boto3_client()

    bucket_name = os.getenv('SDG_OBJECT_STORE_BUCKET')
    s3_key = os.getenv('SDG_OBJECT_STORE_DATA_KEY') # TODO: change the name for the model name
    input_file = '{SDG_PVC_MOUNT_PATH}/sdg.tar.gz' # TODO: change for model path

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

if [[ "$STRATEGY" == "download" ]]; then
  mkdir -p {SDG_PVC_MOUNT_PATH}/generated
  tar -xvf {SDG_PVC_MOUNT_PATH}/sdg.tar.gz -C {SDG_PVC_MOUNT_PATH}/generated
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
    default="standard",
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
@click.option(
    "--namespace", type=str, default="default", help="Kubernetes namespace to use"
)
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
    default="standard",
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
    "--nproc-per-node",
    type=int,
    help="Number of processes per node - for training only",
    default=1,
)
@click.option(
    "--eval-type",
    help="Type of evaluation to run",
    type=click.Choice(["mt-bench"]),
    hidden=True,
)
@click.option(
    "--training-phase",
    help="Type of training phase to run",
    type=click.Choice(["1", "2"]),
)
@click.option(
    "--model-to-train",
    help="Path to model to train (PVC filesystem path)",
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
        "Name of tarball that contains SDG data. (SDG_OBJECT_STORE_DATA_KEY env var)."
        "The tarball MUST NOT contain a top-level directory. "
        "To archive your SDG data, use the following command: cd /path/to/data && tar -czvf sdg.tar.gz *"
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
@click.pass_context
def run(
    ctx: click.Context,
    namespace: typing.Optional[str] = "default",
    taxonomy_repo_url: str = "",
    taxonomy_repo_branch: typing.Optional[str] = "",
    taxonomy_repo_pr: typing.Optional[str] = "",
    storage_class: typing.Optional[str] = "standard",
    serving_endpoint: typing.Optional[str] = None,
    serving_model: typing.Optional[str] = None,
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
        # ctx.invoke(train)

        # Evaluation of phase 2 with MT-Bench
        # ctx.obj["eval_type"] = "mt-bench"
        # _ = ctx.invoke(evaluation)


def get_security_context() -> kubernetes.client.V1SecurityContext:
    """
    Get the security context.
    """
    return kubernetes.client.V1SecurityContext(
        capabilities=kubernetes.client.V1Capabilities(drop=["ALL"]),
        run_as_non_root=True,
    )


def get_sdg_vol_mount() -> kubernetes.client.V1VolumeMount:
    """
    Get the volume mount for the SDG job.
    """
    return [
        kubernetes.client.V1VolumeMount(
            name=SDG_VOLUME_NAME, mount_path=SDG_PVC_MOUNT_PATH
        ),
        kubernetes.client.V1VolumeMount(
            name=MODEL_VOLUME_NAME, mount_path=MODEL_PVC_MOUNT_PATH
        ),
        kubernetes.client.V1VolumeMount(
            name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
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
    init_containers = [
        kubernetes.client.V1Container(
            name="sdg-op-fetch-taxonomy-data",
            image="registry.access.redhat.com/ubi9/toolbox",
            command=["/bin/sh", "-c"],
            args=[
                'git clone {exec_git_clone_op_repo_url} {TAXONOMY_PATH} && cd {TAXONOMY_PATH} && if [ -n "{exec_git_clone_op_repo_branch}" ]; then git fetch origin {exec_git_clone_op_repo_branch} && git checkout {exec_git_clone_op_repo_branch}; elif [ -n "{exec_git_clone_op_repo_pr}" ] && [ {exec_git_clone_op_repo_pr} -gt 0 ]; then git fetch origin pull/{exec_git_clone_op_repo_pr}/head:{exec_git_clone_op_repo_pr} && git checkout {exec_git_clone_op_repo_pr}; fi '
            ],
            volume_mounts=get_sdg_vol_mount(),
            security_context=get_security_context(),
        ),
        kubernetes.client.V1Container(
            name="sdg-op-generate-synthetic-data",
            image="quay.io/tcoufal/ilab-sdg:latest",
            command=[
                "sh",
                "-c",
                '\nif ! [ -x "$(command -v pip)" ]; then\n    python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location \'kfp==2.9.0\' \'--no-deps\' \'typing-extensions>=3.7.4,<5; python_version<"3.9"\' && "$0" "$@"\n',
                "sh",
                "-ec",
                'program_path=$(mktemp -d)\n\nprintf "%s" "$0" > "$program_path/ephemeral_component.py"\n_KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"\n',
                '\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import *\n\ndef sdg_op(\n    num_instructions_to_generate: int,\n    taxonomy: dsl.Input[dsl.Dataset],\n    sdg: dsl.Output[dsl.Dataset],\n    repo_branch: Optional[str],\n    repo_pr: Optional[int],\n):\n    from os import getenv\n\n    import openai\n    from instructlab.sdg import generate_data\n    from instructlab.sdg.utils.taxonomy import read_taxonomy\n\n    api_key = getenv("api_key")\n    model = getenv("model")\n    endpoint = getenv("endpoint")\n    client = openai.OpenAI(base_url=endpoint, api_key=api_key)\n\n    taxonomy_base = "main" if repo_branch or (repo_pr and int(repo_pr) > 0) else "empty"\n\n    print("Generating syntetic dataset for:")\n    print()\n    print(read_taxonomy(taxonomy.path, taxonomy_base))\n\n    # generate_data has a magic word for its taxonomy_base argument - `empty`\n    # it allows generating from the whole repo, see:\n    # https://github.com/instructlab/sdg/blob/c6a9e74a1618b1077cd38e713b8aaed8b7c0c8ce/src/instructlab/sdg/utils/taxonomy.py#L230\n    generate_data(\n        client=client,\n        num_instructions_to_generate=num_instructions_to_generate,\n        output_dir=sdg.path,\n        taxonomy=taxonomy.path,\n        taxonomy_base=taxonomy_base,\n        model_name=model,\n        chunk_word_count=1000,\n        server_ctx_size=4096,\n    )\n\n',
            ],
            args=[
                "--executor_input",
                '{"inputs": {"parameterValues": {"num_instructions_to_generate": 2, "repo_branch": "", "repo_pr": ""}, "artifacts": {"taxonomy": {"artifacts": [{"name": "taxonomy", "uri": "/input_data/taxonomy"}]}}}, "outputs": {"outputFile": "/tmp/kfp_outputs/output_metadata.json", "artifacts": {"sdg": {"artifacts": [{"name": "sdg", "uri": "/input_data/generated"}]}}}}',
                "--function_to_execute",
                "sdg_op",
            ],
            volume_mounts=get_sdg_vol_mount(),
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
            command=[
                "sh",
                "-c",
                "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location 'kfp==2.9.0' '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"'  &&  python3 -m pip install --quiet --no-warn-script-location 'huggingface_hub' && \"$0\" \"$@\"\n",
                "sh",
                "-ec",
                'program_path=$(mktemp -d)\n\nprintf "%s" "$0" > "$program_path/ephemeral_component.py"\n_KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"\n',
                '\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import *\n\ndef huggingface_importer_op(model: dsl.Output[dsl.Model], repo_name: str):\n    from huggingface_hub import snapshot_download\n\n    snapshot_download(repo_id=repo_name, cache_dir="/tmp", local_dir=model.path)\n\n',
            ],
            args=[
                "--executor_input",
                '{"inputs": {"parameterValues": {"repo_name": "ibm-granite/granite-7b-base"}}, "outputs": {"outputFile": "/tmp/kfp_outputs/output_metadata.json", "artifacts": {"model": {"artifacts": [{"name": "model", "uri": "/input_model"}]}}}}',
                "--function_to_execute",
                "huggingface_importer_op",
            ],
            volume_mounts=get_sdg_vol_mount(),
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
            command=[
                "sh",
                "-c",
                "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location 'kfp==2.9.0' '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"'  &&  python3 -m pip install --quiet --no-warn-script-location 'instructlab-training@git+https://github.com/instructlab/training.git' && \"$0\" \"$@\"\n",
                "sh",
                "-ec",
                'program_path=$(mktemp -d)\n\nprintf "%s" "$0" > "$program_path/ephemeral_component.py"\n_KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"\n',
                '\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import *\n\ndef data_processing_op(\n    sdg: dsl.Input[dsl.Dataset],\n    processed_data: dsl.Output[dsl.Dataset],\n    model: dsl.Input[dsl.Artifact],\n    max_seq_len: Optional[int] = 4096,\n    max_batch_len: Optional[int] = 20000,\n):\n    import os\n\n    import instructlab.training.data_process as dp\n    from instructlab.training import (\n        DataProcessArgs,\n        TrainingArgs,\n    )\n\n    # define training-specific arguments\n    training_args = TrainingArgs(\n        # define data-specific arguments\n        model_path=model.path,\n        data_path=f"{sdg.path}/*_train_msgs*.jsonl",\n        data_output_dir=processed_data.path,\n        # define model-trianing parameters\n        max_seq_len=max_seq_len,\n        max_batch_len=max_batch_len,\n        # XXX(shanand): We don\'t need the following arguments\n        # for data processing. Added them for now to avoid\n        # Pydantic validation errors for TrainingArgs\n        ckpt_output_dir="data/saved_checkpoints",\n        num_epochs=2,\n        effective_batch_size=3840,\n        save_samples=0,\n        learning_rate=2e-6,\n        warmup_steps=800,\n        is_padding_free=True,\n    )\n\n    def data_processing(train_args: TrainingArgs) -> None:\n        # early validation logic here\n        if train_args.max_batch_len < train_args.max_seq_len:\n            raise ValueError(\n                f"the `max_batch_len` cannot be less than `max_seq_len`: {train_args.max_batch_len=} < {train_args.max_seq_len=}"\n            )\n\n            # process the training data\n        if not os.path.exists(train_args.data_output_dir):\n            os.makedirs(train_args.data_output_dir, exist_ok=True)\n        dp.main(\n            DataProcessArgs(\n                # XXX(osilkin): make a decision here, either:\n                #   1. the CLI is fully responsible for managing where the data is written\n                #   2. we never cache it and simply write it to a tmp file every time.\n                #\n                # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux\n                # where the user has a defined place for new temporary data to be written.\n                data_output_path=train_args.data_output_dir,\n                model_path=train_args.model_path,\n                data_path=train_args.data_path,\n                max_seq_len=train_args.max_seq_len,\n                chat_tmpl_path=train_args.chat_tmpl_path,\n            )\n        )\n\n    data_processing(train_args=training_args)\n\n',
            ],
            args=[
                "--executor_input",
                '{"inputs": {"parameterValues": {"max_seq_len": 4096, "max_batch_len": 20000}, "artifacts": {"sdg": {"artifacts": [{"name": "sdg", "uri": "/input_data/generated"}]}, "model": {"artifacts": [{"name": "model", "uri": "/input_model"}]}}}, "outputs": {"outputFile": "/tmp/kfp_outputs/output_metadata.json", "artifacts": {"processed_data": {"artifacts": [{"name": "processed_data", "uri": "/input_data/processed_data"}]}}}}',
                "--function_to_execute",
                "data_processing_op",
            ],
            volume_mounts=get_sdg_vol_mount(),
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
        args=[f"cp -r -v {MODEL_PVC_MOUNT_PATH} {TRAINING_PVC_MOUNT_PATH}"],
        volume_mounts=get_sdg_vol_mount(),
    )

    volumes = [
        kubernetes.client.V1Volume(
            name=SDG_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=SDG_PVC_NAME
            ),
        ),
        kubernetes.client.V1Volume(
            name=MODEL_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=MODEL_PVC_NAME
            ),
        ),
        kubernetes.client.V1Volume(
            name=TRAINING_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=TRAINING_PVC_NAME
            ),
        ),
    ]

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

    container = kubernetes.client.V1Container(
        name="fetch-sdg-files-from-object-store",
        image=PYTHON_IMAGE,
        command=["/bin/sh", "-c"],
        args=[
            SDG_DATA_SCRIPT.format(
                strategy="download", SDG_PVC_MOUNT_PATH=SDG_PVC_MOUNT_PATH
            )
        ],
        volume_mounts=get_sdg_vol_mount(),
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
                        name=sdg_object_store_secret, key="access_key", optional=False
                    )
                ),
            ),
            kubernetes.client.V1EnvVar(
                name="SDG_OBJECT_STORE_SECRET_KEY",
                value_from=kubernetes.client.V1EnvVarSource(
                    secret_key_ref=kubernetes.client.V1SecretKeySelector(
                        name=sdg_object_store_secret, key="secret_key", optional=False
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
                        name=sdg_object_store_secret, key="verify_tls", optional=True
                    )
                ),
            ),
        ],
    )

    volumes = [
        kubernetes.client.V1Volume(
            name=SDG_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=SDG_PVC_NAME
            ),
        ),
        kubernetes.client.V1Volume(
            name=MODEL_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=MODEL_PVC_NAME
            ),
        ),
        kubernetes.client.V1Volume(
            name=TRAINING_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=TRAINING_PVC_NAME
            ),
        ),
    ]

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "sdg-data-fetch"}),
        spec=kubernetes.client.V1PodSpec(
            restart_policy="Never",
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


def create_eval_job(
    namespace: str,
    job_name: str,
    eval_type: str,
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
    if eval_type == "mt-bench":
        init_containers = [
            kubernetes.client.V1Container(
                name=f"run-eval-{eval_type}",
                image="quay.io/sallyom/instructlab-ocp:eval",
                command=[
                    "sh",
                    "-c",
                    '\nif ! [ -x "$(command -v pip)" ]; then\n    python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location \'kfp==2.9.0\' \'--no-deps\' \'typing-extensions>=3.7.4,<5; python_version<"3.9"\' && "$0" "$@"\n',
                    "sh",
                    "-ec",
                    'program_path=$(mktemp -d)\n\nprintf "%s" "$0" > "$program_path/ephemeral_component.py"\n_KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"\n',
                    '\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import *\n\ndef run_mmlu_op(\n    mmlu_output: Output[Artifact],\n    models_path_prefix: str,\n    mmlu_tasks_list: str,\n    model_dtype: str,\n    few_shots: int,\n    batch_size: int,\n    device: str = None,\n    models_list: List[str] = None,\n    models_folder: Optional[str] = None,\n) -> NamedTuple("outputs", best_model=str, best_score=float):\n    import json\n    import os\n\n    import torch\n    from instructlab.eval.mmlu import MMLU_TASKS, MMLUEvaluator\n\n    mmlu_tasks = mmlu_tasks_list.split(",") if mmlu_tasks_list else MMLU_TASKS\n\n    if models_list is None and models_folder:\n        models_list = os.listdir(models_folder)\n\n    # Device setup and debug\n    gpu_available = torch.cuda.is_available()\n    gpu_name = (\n        torch.cuda.get_device_name(torch.cuda.current_device())\n        if gpu_available\n        else "No GPU available"\n    )\n\n    print(f"GPU Available: {gpu_available}, Using: {gpu_name}")\n\n    effective_device = (\n        device if device is not None else ("cuda" if gpu_available else "cpu")\n    )\n    print(f"Running on device: {effective_device}")\n\n    scores = {}\n    all_mmlu_data = []\n\n    for model_name in models_list:\n        model_path = f"{models_path_prefix}/{model_name}"\n        # Debug\n        print(f"Model {model_name} is stored at: {model_path}")\n\n        # Evaluation\n        evaluator = MMLUEvaluator(\n            model_path=model_path,\n            tasks=mmlu_tasks,\n            model_dtype=model_dtype,\n            few_shots=few_shots,\n            batch_size=batch_size,\n            device=effective_device,\n        )\n\n        mmlu_score, individual_scores = evaluator.run()\n        average_score = round(mmlu_score, 2)\n        print(\n            f"Model {model_name} is stored at: {model_path} with AVERAGE_SCORE: {average_score}"\n        )\n\n        mmlu_data = {\n            "report_title": "KNOWLEDGE EVALUATION REPORT",\n            "model": model_name,\n            "average_score": average_score,\n            "number_of_tasks": len(individual_scores),\n            "individual_scores": [\n                {task: round(score["score"], 2)}\n                for task, score in individual_scores.items()\n            ],\n        }\n\n        all_mmlu_data.append(mmlu_data)\n        scores[model_path] = average_score\n\n    with open(mmlu_output.path, "w") as f:\n        json.dump(all_mmlu_data, f, indent=4)\n    outputs = NamedTuple("outputs", best_model=str, best_score=float)\n    best_model = max(scores, key=scores.get)\n    best_score = scores[best_model]\n    return outputs(best_model=best_model, best_score=best_score)\n\n',
                ],
                args=[
                    "--executor_input",
                    '{"inputs": {"parameterValues": {"models_path_prefix": "/output/model/hf_format", "mmlu_tasks_list": "mmlu_anatomy,mmlu_astronomy", "model_dtype": "bfloat16", "few_shots": 5, "batch_size": 8, "models_folder": "/output/model/hf_format"}}, "outputs": {"outputFile": "/tmp/kfp_outputs/output_metadata.json", "artifacts": {"mmlu_output": {"artifacts": [{"name": "mmlu_output", "uri": "/output/mmlu-results.txt"}]}}}}',
                    "--function_to_execute",
                    "run_mmlu_op",
                ],
                volume_mounts=[
                    kubernetes.client.V1VolumeMount(
                        name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
                    ),
                ],
            )
        ]
        container = kubernetes.client.V1Container(
            name=f"output-eval-{eval_type}-scores",
            image="quay.io/sallyom/instructlab-ocp:eval",
            command=["/bin/sh", "-c"],
            args=[f"cat {MMLU_SCORES_PATH}"],
            volume_mounts=[
                kubernetes.client.V1VolumeMount(
                    name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
                ),
            ],
        )
    elif eval_type == "mt-bench":
        init_containers = [
            kubernetes.client.V1Container(
                name=f"run-eval-{eval_type}",
                image="quay.io/sallyom/instructlab-ocp:eval-7ee213",
                command=[
                    "sh",
                    "-c",
                    "\nif ! [ -x \"$(command -v pip)\" ]; then\n    python3 -m ensurepip || python3 -m ensurepip --user || apt-get install python3-pip\nfi\n\nPIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --quiet --no-warn-script-location 'kfp==2.9.0' '--no-deps' 'typing-extensions>=3.7.4,<5; python_version<\"3.9\"'  &&  python3 -m pip install --quiet --no-warn-script-location 'vllm' && \"$0\" \"$@\"\n",
                    "sh",
                    "-ec",
                    'program_path=$(mktemp -d)\n\nprintf "%s" "$0" > "$program_path/ephemeral_component.py"\n_KFP_RUNTIME=true python3 -m kfp.dsl.executor_main                         --component_module_path                         "$program_path/ephemeral_component.py"                         "$@"\n',
                    '\nimport kfp\nfrom kfp import dsl\nfrom kfp.dsl import *\nfrom typing import *\n\ndef run_mt_bench_op(\n    models_path_prefix: str,\n    mt_bench_output: Output[Artifact],\n    merge_system_user_message: bool,\n    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - `auto`\n    # with `auto`, number of gpus allocated for serving is calculated based on environment\n    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36\n    max_workers: str,\n    models_list: List[str] = None,\n    models_folder: Optional[str] = None,\n    device: str = None,\n) -> NamedTuple("outputs", best_model=str, best_score=float):\n    def launch_vllm(model_path: str, gpu_count: int, retries: int = 60, delay: int = 5):\n        import subprocess\n        import sys\n        import time\n\n        import requests\n\n        if gpu_count > 0:\n            command = [\n                sys.executable,\n                "-m",\n                "vllm.entrypoints.openai.api_server",\n                "--model",\n                model_path,\n                "--tensor-parallel-size",\n                str(gpu_count),\n            ]\n        else:\n            command = [\n                sys.executable,\n                "-m",\n                "vllm.entrypoints.openai.api_server",\n                "--model",\n                model_path,\n            ]\n\n        subprocess.Popen(args=command)\n\n        server_url = "http://localhost:8000/v1"\n        print(f"Waiting for vLLM server to start at {server_url}...")\n\n        for attempt in range(retries):\n            try:\n                response = requests.get(f"{server_url}/models")\n                if response.status_code == 200:\n                    print(f"vLLM server is up and running at {server_url}.")\n                    return\n            except requests.ConnectionError:\n                pass\n\n            print(\n                f"Server not available yet, retrying in {delay} seconds (Attempt {attempt + 1}/{retries})..."\n            )\n            time.sleep(delay)\n\n        raise RuntimeError(\n            f"Failed to start vLLM server at {server_url} after {retries} retries."\n        )\n\n    # This seems like excessive effort to stop the vllm process, but merely saving & killing the pid doesn\'t work\n    # Also, the base image does not include `pkill` cmd, so can\'t pkill -f vllm.entrypoints.openai.api_server either\n    def stop_vllm_server_by_name():\n        import psutil\n\n        for process in psutil.process_iter(attrs=["pid", "name", "cmdline"]):\n            cmdline = process.info.get("cmdline")\n            if cmdline and "vllm.entrypoints.openai.api_server" in cmdline:\n                print(\n                    f"Found vLLM server process with PID: {process.info[\'pid\']}, terminating..."\n                )\n                try:\n                    process.terminate()  # Try graceful termination\n                    process.wait(timeout=5)  # Wait a bit for it to terminate\n                    if process.is_running():\n                        print(\n                            f"Forcefully killing vLLM server process with PID: {process.info[\'pid\']}"\n                        )\n                        process.kill()  # Force kill if it\'s still running\n                    print(\n                        f"Successfully stopped vLLM server with PID: {process.info[\'pid\']}"\n                    )\n                except psutil.NoSuchProcess:\n                    print(f"Process with PID {process.info[\'pid\']} no longer exists.")\n                except psutil.AccessDenied:\n                    print(\n                        f"Access denied when trying to terminate process with PID {process.info[\'pid\']}."\n                    )\n                except Exception as e:\n                    print(\n                        f"Failed to terminate process with PID {process.info[\'pid\']}. Error: {e}"\n                    )\n\n    import json\n    import os\n\n    import torch\n    from instructlab.eval.mt_bench import MTBenchEvaluator\n\n    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"\n    vllm_server = "http://localhost:8000/v1"\n\n    gpu_available = torch.cuda.is_available()\n    gpu_name = (\n        torch.cuda.get_device_name(torch.cuda.current_device())\n        if gpu_available\n        else "No GPU available"\n    )\n    gpu_count = torch.cuda.device_count() if gpu_available else 0\n\n    print(f"GPU Available: {gpu_available}, {gpu_name}")\n\n    if models_list is None and models_folder:\n        models_list = os.listdir(models_folder)\n\n    judge_api_key = os.getenv("JUDGE_API_KEY", "")\n    judge_model_name = os.getenv("JUDGE_NAME")\n    judge_endpoint = os.getenv("JUDGE_ENDPOINT")\n\n    scores = {}\n    all_mt_bench_data = []\n\n    # generate_answers,judgment uses a magic word for its mt_bench evaluator  - `auto`\n    # with `auto`, number of gpus allocated for serving is calculated based on environment\n    # https://github.com/instructlab/eval/blob/main/src/instructlab/eval/mt_bench.py#L36\n    if max_workers == "auto":\n        try:\n            usable_cpu_count = len(os.sched_getaffinity(0)) // 2\n        except AttributeError:\n            usable_cpu_count = multiprocessing.cpu_count() // 2\n        max_workers = usable_cpu_count\n\n    for model_name in models_list:\n        print(f"Serving candidate model: {model_name}")\n        model_path = f"{models_path_prefix}/{model_name}"\n\n        launch_vllm(model_path, gpu_count)\n\n        # model ID is the model_path value in vLLM\n        evaluator = MTBenchEvaluator(\n            model_name=model_path,\n            judge_model_name=judge_model_name,\n            output_dir="/tmp/eval_output",\n            merge_system_user_message=merge_system_user_message,\n        )\n\n        evaluator.gen_answers(\n            server_url=vllm_server,\n            serving_gpus=gpu_count,\n            max_workers=max_workers,\n        )\n\n        stop_vllm_server_by_name()\n\n        overall_score, qa_pairs, turn_scores, error_rate = evaluator.judge_answers(\n            server_url=judge_endpoint,\n            api_key=judge_api_key,\n            serving_gpus=gpu_count,\n            max_workers=max_workers,\n        )\n\n        mt_bench_data = {\n            "report_title": "SKILLS EVALUATION REPORT",\n            "model": model_path,\n            "judge_model": judge_model_name,\n            "overall_score": overall_score,\n            "turn_scores": turn_scores,\n            "qa_scores": qa_pairs,\n            "error_rate": error_rate,\n        }\n\n        all_mt_bench_data.append(mt_bench_data)\n        scores[model_path] = overall_score\n\n    with open(mt_bench_output.path, "w") as f:\n        json.dump(all_mt_bench_data, f, indent=4)\n\n    outputs = NamedTuple("outputs", best_model=str, best_score=float)\n    best_model = max(scores, key=scores.get)\n    best_score = scores[best_model]\n    return outputs(best_model=best_model, best_score=best_score)\n\n',
                ],
                args=[
                    "--executor_input",
                    '{"inputs": {"parameterValues": {"models_path_prefix": "/output/model/hf_format", "merge_system_user_message": false, "max_workers": "auto"}}, "outputs": {"outputFile": "/tmp/kfp_outputs/output_metadata.json", "artifacts": {"mt_bench_output": {"artifacts": [{"name": "mt_bench_output", "uri": "/output/mt-bench-results.txt"}]}}}}',
                    "--function_to_execute",
                    "run_mt_bench_op",
                ],
                volume_mounts=[
                    kubernetes.client.V1VolumeMount(
                        name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
                    ),
                ],
            )
        ]
        container = kubernetes.client.V1Container(
            name=f"output-eval-{eval_type}-scores",
            image="quay.io/sallyom/instructlab-ocp:eval-7ee213",
            command=["/bin/sh", "-c"],
            args=[f"cat {MT_BENCH_SCORES_PATH}"],
            volume_mounts=[
                kubernetes.client.V1VolumeMount(
                    name=TRAINING_VOLUME_NAME, mount_path=TRAINING_PVC_MOUNT_PATH
                ),
            ],
        )
    else:
        raise ValueError(f"Unknown evaluation type: {eval_type}")

    volumes = [
        kubernetes.client.V1Volume(
            name=TRAINING_VOLUME_NAME,
            persistent_volume_claim=kubernetes.client.V1PersistentVolumeClaimVolumeSource(
                claim_name=TRAINING_PVC_NAME
            ),
        ),
    ]

    # Create and configure a spec section
    template = kubernetes.client.V1PodTemplateSpec(
        metadata=kubernetes.client.V1ObjectMeta(labels={"app": "eval"}),
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
            "name": SDG_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteOnce"],
            "size": "1Gi",
        },
        {
            "name": MODEL_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteOnce"],
            "size": "50Gi",
        },
        {
            "name": TRAINING_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteMany"],
            "size": "50Gi",
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
    sdg_object_store_endpoint = ctx.obj["sdg_object_store_endpoint"]
    sdg_object_store_bucket = ctx.obj["sdg_object_store_bucket"]
    sdg_object_store_access_key = ctx.obj["sdg_object_store_access_key"]
    sdg_object_store_secret_key = ctx.obj["sdg_object_store_secret_key"]
    sdg_object_store_region = ctx.obj["sdg_object_store_region"]
    sdg_object_store_data_key = ctx.obj["sdg_object_store_data_key"]
    sdg_object_store_verify_tls = ctx.obj["sdg_object_store_verify_tls"]
    sdg_object_store_secret = ctx.obj["sdg_object_store_secret"]

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

    # list of PVCs to create and their details
    pvcs = [
        {
            "name": SDG_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteOnce"],
            "size": "1Gi",
        },
        {
            "name": MODEL_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteOnce"],
            "size": "50Gi",
        },
        {
            "name": TRAINING_PVC_NAME,
            "namespace": namespace,
            "storage_class": storage_class,
            "access_modes": ["ReadWriteMany"],
            "size": "50Gi",
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

    if training_phase is None:
        raise ValueError("Training phase must be provided with --training-phase=[1|2]")

    # During the initial training
    if path_to_model is None:
        path_to_model = "/input_model"

    logger.info("Running multi-phased distributed training phase %s", training_phase)
    worker_replicas = PYTORCH_NNODES - 1
    pytorch_training_job_yaml = yaml.safe_load(
        PYTORCH_TRAINING_JOB.format(
            name="train-sdg",
            model_pvc_name="model",
            input_pvc_name="sdg-data",
            output_pvc_name="training-data",
            path_to_model=path_to_model,
            nproc_per_node=nproc_per_node,
            PYTORCH_NNODES=PYTORCH_NNODES,
            PYTORCH_IMAGE=PYTORCH_IMAGE,
            worker_replicas=worker_replicas,
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
        if isinstance(scores_data, list):
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
