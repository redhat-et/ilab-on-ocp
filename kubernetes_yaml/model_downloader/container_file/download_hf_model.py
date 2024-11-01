import logging
import os
import shutil
from pathlib import Path
from typing import Union

import boto3
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)


def upload_and_save_model_to_s3(
    model_name: str,
    local_model_path: Union[str, Path],
    s3_model_path: str,
    verbose: bool = False,
    replace_if_exists: bool = False,
) -> str:
    """
    Download a Hugging Face model, upload it to S3, and log the details.

    Args:
        model_name (str): Model name/ID from Hugging Face.
        local_model_path (Union[str, Path]): Local model directory path.
        s3_model_path (str): S3 path to store the model.
        verbose (bool): If True, show a progress bar.
        replace_if_exists (bool): If True, re-download the model if it exists locally.

    Returns:
        str: S3 path where the model is stored.
    """
    # Convert s3_model_path to lowercase
    s3_model_path = os.getenv("MODEL_PATH")
    s3_model_path = s3_model_path.lower()

    # S3 Configuration
    s3_endpoint = os.getenv("S3_ENDPOINT")
    aws_access_key = os.getenv("AWS_ACCESS_KEY")
    aws_secret_key = os.getenv("AWS_SECRET_KEY")
    bucket_name = os.getenv("AWS_BUCKET_NAME")
    s3_folder = os.getenv("S3_FOLDER")
    s3_prefix = f"{s3_folder}/{s3_model_path}"

    # Create an S3 client
    s3 = boto3.client(
        "s3",
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key,
    )

    # Handle Hugging Face model download
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        hf_token = hf_token.strip()
        login(token=hf_token, add_to_git_credential=True)
        logger.info("Successfully logged in to Hugging Face.")
    else:
        raise EnvironmentError(
            "HF_TOKEN is not defined. Please set the Hugging Face token as an environment variable."
        )

    converted_model_path = os.path.join(local_model_path, model_name.replace("/", "-"))

    if Path(converted_model_path).exists() and not replace_if_exists:
        logger.info(f"Path '{converted_model_path}' already exists. Skipping download.")
    else:
        if Path(converted_model_path).exists():
            shutil.rmtree(converted_model_path)
        snapshot_download(repo_id=model_name, local_dir=converted_model_path)
        logger.info(
            f"Model '{model_name}' downloaded and saved to {converted_model_path}"
        )

    # Upload files to S3 with a progress bar
    total_files = sum(len(files) for _, _, files in os.walk(converted_model_path))
    num_files = 0
    with tqdm(
        total=total_files, desc="Uploading files to S3", disable=not verbose
    ) as pbar:
        for root, _, files in os.walk(converted_model_path):
            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, converted_model_path)
                s3_key = os.path.join(s3_prefix, relative_path)
                logger.info(f"Uploading {file_path} to s3://{bucket_name}/{s3_key}")
                s3.upload_file(file_path, bucket_name, s3_key)
                num_files += 1
                pbar.update(1)

    if num_files == 0:
        raise ValueError(
            f"No files were uploaded. Check access to {converted_model_path}."
        )

    # Log connection details
    s3_path = f"s3://{bucket_name}/{s3_prefix}"
    message = (
        f"\nModel ready to be added as a data connection on Openshift AI with following parameters:\n"
        f"Name: <preferred-name>\n"
        f"Access key: <secret>\n"
        f"Secret key: <secret>\n"
        f"Endpoint: https://s3.us-east-1.amazonaws.com\n"
        f"Region: us-east-1\n"
        f"Bucket: {bucket_name}\n"
        f"Path: {s3_path}\n"
    )
    logger.info(message)

    return s3_path


# Example usage
upload_and_save_model_to_s3(
    model_name=os.getenv("MODEL"),
    local_model_path="./models",
    s3_model_path=os.getenv("MODEL_PATH"),
    verbose=True,
    replace_if_exists=False,
)
