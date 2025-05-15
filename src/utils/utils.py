import yaml
from urllib.parse import urlparse
import pandas as pd
import os

def load_config(config_path: str = "cfg/config.yaml") -> dict:
    """
    Load configuration from a YAML file.

    Args:
        config_path (str): The path to the configuration YAML file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def extract_s3_path_from_url(url: str) -> str:
    """
    Extract the S3 object path from an S3 URL or URI.

    This function parses S3 URLs/URIs and returns just the object path portion,
    removing the protocol (s3://), bucket name, and any leading slashes.

    Args:
        url (str): The full S3 URI (e.g., 's3://eodata/path/to/file.jp2')

    Returns:
        str: The S3 object path (without protocol, bucket name and leading slashes)

    Raises:
        ValueError: If the provided URL is not an S3 URL.
    """
    if not url.startswith('s3://'):
        return url

    parsed_url = urlparse(url)

    if parsed_url.scheme != 's3':
        raise ValueError(f"URL {url} is not an S3 URL")

    object_path = parsed_url.path.lstrip('/')
    return object_path


def prepare_paths(path_dir: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Prepare paths for input and output datasets from CSV files.

    Args:
        path_dir (str): Directory containing input and target CSV files.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames for input and output datasets.
    """
    df_input = pd.read_csv(os.path.join(path_dir, "input.csv"))
    df_output = pd.read_csv(os.path.join(path_dir, "target.csv"))

    df_input["path"] = df_input["Name"].apply(
        lambda x: os.path.join(path_dir, "input", os.path.basename(x).replace(".SAFE", ""))
    )
    df_output["path"] = df_output["Name"].apply(
        lambda x: os.path.join(path_dir, "target", os.path.basename(x).replace(".SAFE", ""))
    )

    return df_input, df_output


def remove_last_segment_rsplit(sentinel_id: str) -> str:
    """
    Remove the last segment from a Sentinel ID by splitting at the last underscore.

    Args:
        sentinel_id (str): The Sentinel ID to process.

    Returns:
        str: The Sentinel ID without the last segment.
    """
    parts = sentinel_id.rsplit('_', 1)
    return parts[0]
