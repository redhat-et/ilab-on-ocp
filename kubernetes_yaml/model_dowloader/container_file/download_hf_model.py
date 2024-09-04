from typing import Union
from pathlib import Path
from huggingface_hub import snapshot_download, login
import shutil
import logging
import os
import boto3
import warnings
from tqdm import tqdm  # Import tqdm for progress bar

logger = logging.getLogger(__name__)
from dotenv import load_dotenv
load_dotenv()

def _path_exists(path: Union[str, Path]):
    return os.path.isdir(path)

def upload_directory_to_s3(local_directory: str, bucket: str, s3_prefix: str, verbose: bool = False) -> int:
    """
    Upload all files in a local directory to a directory of the same name in s3 with a progress bar.

    Args:
        local_directory (str): Path to the local directory to upload to S3.
        bucket (str): Bucket to upload to.
        s3_prefix (str): Path within the bucket to upload to.
        verbose (bool): If True, display a progress bar.

    Returns:
        int: Number of files successfully uploaded.
    """

    s3_endpoint = os.environ.get('S3_ENDPOINT')
    aws_access_key = os.environ.get('AWS_ACCESS_KEY')
    aws_secret_key = os.environ.get('AWS_SECRET_KEY')

    # Create an S3 client
    s3 = boto3.client(
        's3',
        endpoint_url=s3_endpoint,
        aws_access_key_id=aws_access_key,
        aws_secret_access_key=aws_secret_key
    )

    # Count total files for the progress bar
    total_files = sum([len(files) for _, _, files in os.walk(local_directory)])

    num_files = 0
    with tqdm(total=total_files, desc="Uploading files to S3", disable=not verbose) as pbar:
        for root, dirs, files in os.walk(local_directory):
            for filename in files:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, local_directory)
                s3_key = os.path.join(s3_prefix, relative_path)
                logger.info(f"{file_path} -> s3://{bucket}/{s3_key}")
                s3.upload_file(file_path, bucket, s3_key)
                num_files += 1
                pbar.update(1)  # Update progress bar

    return num_files

def save_model_to_s3(local_model_path: Union[str, Path], s3_model_path: str, verbose: bool = False) -> str:
    """Save a model directory to s3 with verbosity and a progress bar."""

    # Convert s3_model_path to lowercase
    s3_model_path = s3_model_path.lower()

    logger.info(f"Connecting to the s3 bucket to upload the model files")
    bucket_name = os.environ.get("AWS_BUCKET_NAME")
    s3_folder = os.environ.get("S3_FOLDER")

    # Push the local folder to S3
    s3_prefix = f"{s3_folder}/{s3_model_path}"
    num_files = upload_directory_to_s3(local_directory=local_model_path, bucket=bucket_name, s3_prefix=s3_prefix, verbose=verbose)
    if num_files == 0:
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

# Example usage with verbosity
## save_model_to_s3(
##    local_model_path="./models",
##    s3_model_path="mixtral-8x7b-v0.1",  # Ensure the path is in lowercase
##    verbose=True  # Enable progress bar
##)

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

    # Check if the Hugging Face token is defined
    hf_token = os.environ.get('HF_TOKEN')

    if hf_token:
        # Strip any whitespace or newlines from the token
        hf_token = hf_token.strip()

        # Log in to Hugging Face using the token and add to Git credentials if necessary
        login(token=hf_token, add_to_git_credential=True)
        logger.info("Successfully logged in to Hugging Face using the provided token.")
    else:
        raise EnvironmentError("HF_TOKEN is not defined. Please set the Hugging Face token as an environment variable.")

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

# Example usage
save_hf_model(
    model_name="mistralai/Mixtral-8x7B-v0.1",
    local_dir="./models",
    s3_model_path="models",
    replace_if_exists=False
)
