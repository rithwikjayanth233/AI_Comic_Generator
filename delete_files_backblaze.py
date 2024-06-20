import os
from b2sdk.v1 import InMemoryAccountInfo, B2Api

def load_backblaze_credentials():
    # Load credentials from files securely
    with open('/path/to/backblaze_id.txt', 'r') as file:
        backblaze_id = file.read().strip()  # Read and strip any extra whitespace/newlines
    with open('/path/to/backblaze.txt', 'r') as file:
        backblaze_key = file.read().strip()  # Read and strip any extra whitespace/newlines
    return backblaze_id, backblaze_key


def authorize_backblaze():
    # Load credentials
    backblaze_id, backblaze_key = load_backblaze_credentials()
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", backblaze_id, backblaze_key)
    return b2_api

def delete_files(bucket_name, prefix_filter):
    b2_api = authorize_backblaze()
    bucket = b2_api.get_bucket_by_name(bucket_name)

    # First, list all possible subfolders
    all_folders = set()
    for file_version, folder_name in bucket.ls(folder_to_list='', show_versions=False):
        folder_prefix = file_version.file_name.split('/')[0] + '/'
        if folder_prefix.startswith(prefix_filter) and folder_prefix not in all_folders:
            all_folders.add(folder_prefix)

    # Now, process each matching folder
    for prefix in all_folders:
        print(f"Processing subfolder: {prefix}")
        files_found = 0
        files_deleted = 0

        # List and delete files with the specified conditions
        for file_version, folder_name in bucket.ls(folder_to_list=prefix, show_versions=True):
            file_name = file_version.file_name
            if file_name.endswith('_new.png') or 'story_' in file_name:
                files_found += 1
                try:
                    b2_api.delete_file_version(file_version.id_, file_name)
                    print(f"Deleted: {file_name}")
                    files_deleted += 1
                except Exception as e:
                    print(f"Failed to delete {file_name}: {e}")

        print(f"Total files found in {prefix}: {files_found}")
        print(f"Total files deleted in {prefix}: {files_deleted}")

if __name__ == "__main__":
    bucket_name = '(BUCKET NAME)'  # Your bucket name like dream_tails
    prefix_filter = '(BUCKET DIRECTORY)'  # Prefix to match subfolders like supercat_volume_, space_tails_
    delete_files(bucket_name, prefix_filter)
