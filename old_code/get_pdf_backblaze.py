import os
from b2sdk.v1 import InMemoryAccountInfo, B2Api, DownloadDestLocalFile
from dotenv import load_dotenv

load_dotenv()

# Load Backblaze credentials from files
with open('/path/to/backblaze.txt', 'r') as backblaze:
    backblaze_key = backblaze.read().strip()
with open('/path/to/backblaze_id.txt', 'r') as backblaze:
    backblaze_id = backblaze.read().strip()

# Authorize Backblaze account
info = InMemoryAccountInfo()
b2_api = B2Api(info)
b2_api.authorize_account("production", backblaze_id, backblaze_key)

# Number of volumes to process
num_volumes = 20  # Specify the number of volumes

for volume_number in range(num_volumes):
    print(f"Processing space_tails_{volume_number}")

    # Set bucket directory for current volume
    bucket_name = 'dream-tails'
    bucket_directory = f'space_tails_{volume_number}/'

    # Download and combine text files from Backblaze
    combined_text = []
    page_num = 0
    while True:
        txt_filename = f"{bucket_directory}page_{page_num}.txt"
        download_path = f"page_{page_num}.txt"  # File will be saved locally
        try:
            bucket = b2_api.get_bucket_by_name(bucket_name)
            download_dest = DownloadDestLocalFile(download_path)
            bucket.download_file_by_name(txt_filename, download_dest)
            with open(download_path, 'r') as txt_file:
                combined_text.append(txt_file.read().strip())
            page_num += 1
        except Exception as e:
            print(f"Finished processing {page_num} pages.")
            break

    # Save combined text to a single file
    combined_txt_path = f'output_story_combined_volume_{volume_number}.txt'
    with open(combined_txt_path, 'w') as combined_file:
        combined_file.write(" ".join(combined_text))

    print(f"Combined text for volume {volume_number} saved to {combined_txt_path}")
