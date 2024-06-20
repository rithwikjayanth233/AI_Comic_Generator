import os
from b2sdk.v1 import InMemoryAccountInfo, B2Api, UploadSourceLocalFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load Backblaze credentials from files
def load_backblaze_credentials():
    with open('/home/rjayanth/StoryDiffusion/backblaze.txt', 'r') as backblaze:
        backblaze_key = backblaze.read().strip()
    with open('/home/rjayanth/StoryDiffusion/backblaze_id.txt', 'r') as backblaze:
        backblaze_id = backblaze.read().strip()
    return backblaze_id, backblaze_key

backblaze_id, backblaze_key = load_backblaze_credentials()

# Authorize Backblaze account
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", backblaze_id, backblaze_key)

# Function to upload a PDF to Backblaze
def upload_pdf_to_backblaze(b2_api, bucket_name, local_file_path, remote_file_path):
    bucket = b2_api.get_bucket_by_name(bucket_name)
    source = UploadSourceLocalFile(local_file_path)
    bucket.upload(source, remote_file_path)
    print(f"Uploaded {local_file_path} to {bucket_name}/{remote_file_path}")

def main():
    base_dir = '/home/rjayanth/StoryDiffusion/outputs'
    bucket_name = 'dream-tails'

    # Process each generated PDF
    for volume_number in range(20):  # Adjust the range as needed
        local_file_path = f"{base_dir}/story_{volume_number}.pdf"
        remote_file_path = f'space_tails_{volume_number}/story_{volume_number}.pdf'

        if os.path.exists(local_file_path):
            print(f"Uploading {local_file_path}...")
            upload_pdf_to_backblaze(b2_api, bucket_name, local_file_path, remote_file_path)
        else:
            print(f"File {local_file_path} does not exist and will be skipped.")

if __name__ == "__main__":
    main()
