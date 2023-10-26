import boto3
from pathlib import Path
from botocore import UNSIGNED
from botocore.client import Config
import os

def get_file_folders(s3_client, bucket_name, prefix=""):
    file_names = []
    folders = []

    default_kwargs = {
        "Bucket": bucket_name,
        "Prefix": prefix
    }
    next_token = ""

    while next_token is not None:
        updated_kwargs = default_kwargs.copy()
        if next_token != "":
            updated_kwargs["ContinuationToken"] = next_token

        response = s3_client.list_objects_v2(**updated_kwargs)
        contents = response.get("Contents")

        for result in contents:
            key = result.get("Key")
            if key[-1] == "/":
                folders.append(key)
            else:
                file_names.append(key)

        next_token = response.get("NextContinuationToken")

    return file_names, folders



def download_files(s3_client, bucket_name, file_names, folders):
    local_path = Path(os.getcwd())

    for folder in folders:
        print("creating directory" + folder)
        folder_path = Path.joinpath(local_path, folder)
				# Create all folders in the path
        folder_path.mkdir(parents=True, exist_ok=True)

    for file_name in file_names:
        print("downloading file " + file_name)
        file_path = Path.joinpath(local_path,"content", file_name)
		# Create folder for parent directory
        file_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(
            bucket_name,
            file_name,
            str(file_path)
        )

client = boto3.client('s3', config=Config(signature_version=UNSIGNED))
file_names, folders = get_file_folders(client, 'eyes-on-the-ground')

print("Downloading Images")
download_files(
        client,
        "eyes-on-the-ground",
        file_names,
        folders
    )
