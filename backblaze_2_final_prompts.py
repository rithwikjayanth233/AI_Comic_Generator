import os
import openai
from b2sdk.v1 import InMemoryAccountInfo, B2Api, DownloadDestLocalFile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load OpenAI API key from a text file
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Path to the file containing the OpenAI API key
api_key_file = '/home/rjayanth/StoryDiffusion/gpt_key.txt'
api_key = load_api_key(api_key_file)

# Initialize OpenAI client with the loaded API key
openai.api_key = api_key

# Function to generate text prompts using ChatGPT API
def generate_prompts(story_text):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates text prompts for a text-to-image diffusion model. Ensure the prompts generated are detailed and simple for the text-to-image diffusion model. Each prompt should start with the main object or character followed by a small description in brackets."},
            {"role": "user", "content": f"Generate a general prompt and a series of detailed prompts from the following story: {story_text}. Ensure the prompts are suitable for a text-to-image diffusion model. Split the general prompt and detailed prompts into two sections."}
        ]
    )
    return response.choices[0].message['content']

# Function to read story text from a file
def read_story(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Function to save prompts to a text file
def save_prompts(prompts, output_path):
    with open(output_path, 'w') as file:
        file.write(prompts)

# Function to download and combine text files from Backblaze
def download_and_combine_text_files(b2_api, bucket_name, bucket_directory, output_path):
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
    with open(output_path, 'w') as combined_file:
        combined_file.write(" ".join(combined_text))
    print(f"Combined text saved to {output_path}")

# Main function
def main():
    base_dir = '/home/rjayanth/StoryDiffusion/backblaze_prompts'
    output_dir = '/home/rjayanth/StoryDiffusion/generated_prompts'
    os.makedirs(output_dir, exist_ok=True)

    # Load Backblaze credentials
    with open('/home/rjayanth/StoryDiffusion/backblaze.txt', 'r') as backblaze:
        backblaze_key = backblaze.read().strip()
    with open('/home/rjayanth/StoryDiffusion/backblaze_id.txt', 'r') as backblaze:
        backblaze_id = backblaze.read().strip()

    # Authorize Backblaze account
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account("production", backblaze_id, backblaze_key)

    # Process each volume
    for volume_number in range(20):  # Adjust the range as needed
        bucket_name = 'dream-tails'
        bucket_directory = f'space_tails_{volume_number}/'
        combined_txt_path = f"{base_dir}/output_story_combined_volume_{volume_number}.txt"
        output_file_path = f"{output_dir}/generated_prompts_volume_{volume_number}.txt"

        print(f"Processing volume {volume_number}...")

        # Download and combine text files
        download_and_combine_text_files(b2_api, bucket_name, bucket_directory, combined_txt_path)

        # Read the story
        story_text = read_story(combined_txt_path)

        # Generate prompts
        prompts = generate_prompts(story_text)

        # Save prompts to a text file
        save_prompts(prompts, output_file_path)

        print(f"Prompts for volume {volume_number} saved to {output_file_path}")

if __name__ == "__main__":
    main()
