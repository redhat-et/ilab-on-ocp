import logging
import os
import shutil
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download

# Load environment variables
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def save_model_locally(
    model_name: str,
    local_model_path: Union[str, Path],
    verbose: bool = False,
    replace_if_exists: bool = False,
) -> str:
    """
    Download a Hugging Face model and save it to the specified local directory.

    Args:
        model_name (str): Model name/ID from Hugging Face.
        local_model_path (Union[str, Path]): Local model directory path.
        verbose (bool): If True, show a progress bar.
        replace_if_exists (bool): If True, re-download the model if it exists locally.

    Returns:
        str: Local path where the model is stored.
    """
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

    converted_model_path = Path(local_model_path) / model_name.replace("/", "-")

    if converted_model_path.exists() and not replace_if_exists:
        logger.info(f"Path '{converted_model_path}' already exists. Skipping download.")
    else:
        if converted_model_path.exists():
            shutil.rmtree(converted_model_path)
        snapshot_download(repo_id=model_name, local_dir=converted_model_path)
        logger.info(
            f"Model '{model_name}' downloaded and saved to {converted_model_path}"
        )

    logger.info(f"Model saved locally at: {converted_model_path}")
    return str(converted_model_path)

# Example usage
save_model_locally(
    model_name=os.getenv("MODEL"),
    local_model_path="/mnt",
    verbose=True,
    replace_if_exists=False,
)
