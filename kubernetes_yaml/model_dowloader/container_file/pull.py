from typing import Union
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub import login
import shutil
import logging
import os
import boto3
import warnings
#from rhs3.client import rhs3_client
#from rhs3.s3 import upload_file

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

def _path_exists(path: Union[str, Path]):
    return os.path.isdir(path)

def upload_directory_to_s3(local_directory: str, bucket: str, s3_prefix: str) -> int:
    """
    Upload all files in a local directory to a directory of the same name in s3.

    Args:
        local_directory (str):
            Path to the local directory to upload to S3
        bucket (str):
            Bucket to upload to
        s3_prefix (str):
            Path within the bucket to upload to

    Returns:
        int:
            Number of files successfully uploaded
    """

    s3_endpoint = os.environ.get('S3_ENDPOINT')
    aws_access_key = os.environ.get('AWS_ACCESS_KEY')
    aws_secret_key = os.environ.get('AWS_SECRET_KEY')

    #client = rhs3_client()

    # Create an S3 client
    s3 = boto3.client(
    's3',
    endpoint_url=s3_endpoint,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
    # region_name='us-east-1'  # Specify the region if needed
    )

    num_files = 0
    for root, dirs, files in os.walk(local_directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, local_directory)
            s3_key = os.path.join(s3_prefix, relative_path)
            logger.info(f"{file_path} -> s3://{bucket}/{s3_key}")
            s3.upload_file(file_path, bucket, s3_key)
            num_files += 1

    return num_files

def save_model_to_s3(local_model_path: Union[str, Path], s3_model_path: str):
    """Save a model directry to s3."""

    logger.info(f"Connecting to the s3 bucket to upload the model files")
    bucket_name = os.environ.get("AWS_BUCKET_NAME")
    s3_folder = os.environ.get("S3_FOLDER")

    # Push the local folder to S3
    s3_prefix = f"{s3_folder}/{s3_model_path}"
    num_files = upload_directory_to_s3(local_directory=local_model_path, bucket=bucket_name, s3_prefix=s3_prefix)
    if num_files == 0:
        # TODO: Figure out what would cause this
        raise ValueError(f"The files were not uploaded. Please confirm that you have read & write access to {local_model_path}.")

    # Log connection details
    s3_path = f"s3://{bucket_name}/{s3_prefix}"
    message = "\nModel ready to be added as a data connection on Openshift AI with following parameters:\n"
    message += "Name: <preferred-name>\n"
    message += "Access key: <secret>\n"
    message += "Secret key: <secret>\n"
    message += "Endpoint: https://s3.us-east-1.amazonaws.com\n"
    message += "Region: us-east-1\n"
    message += f"Bucket: {bucket_name}\n"
    message += f"Path: {s3_path}\n"

    logger.info(message)

    return s3_path

def save_hf_model(model_name: str, local_dir: Union[str, Path], s3_model_path: str, replace_if_exists: bool = False):
    """
    Save a HuggingFace model to S3 for model serving.

    Args:
        model_name (str):
            Model name/ID from HuggingFace.
        local_dir (str or Path):
            Path to a local directory to store converted models for upload to S3.
        s3_model_path (str):
            Path to store the model in S3.
        replace_if_exists (bool):
            When False, this will not attempt to redownload the model before uploading to S3 if the local model path already exists.
            Otherwise, it will remove the local model path and redownload the model each time this is run.

    Returns:
        str:
            Path to the model files in S3.
    """

    # Login to HuggingFace to download models using your account token
    hf_token = os.environ.get('HF_TOKEN')
    login(hf_token)
    # Download the safetensors from HuggingFace
    os.environ["ALLOW_DOWNLOADS"] = "1"
    model_path_name = model_name.replace("/", "-")
    converted_model_path = f"{local_dir}/{model_path_name}"
    if _path_exists(converted_model_path) and not replace_if_exists:
        warnings.warn(f"Path '{converted_model_path}' already exists. Download from HF will be skipped.")
    else:
        if _path_exists(converted_model_path):
            shutil.rmtree(path=converted_model_path)
        snapshot_download(repo_id=model_name, local_dir=converted_model_path)
        logger.info(f"Model saved to {converted_model_path}")

    # Upload the model files to S3
    s3_path = save_model_to_s3(local_model_path=converted_model_path, s3_model_path=s3_model_path)

    return s3_path

save_hf_model(model_name="mistralai/Mixtral-8x7B-v0.1",local_dir="./models",s3_model_path="models",replace_if_exists=False)
