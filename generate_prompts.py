import os
import openai

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
            {"role": "system", "content": "You are a helpful assistant that generates text prompts for a text-to-image diffusion model. Ensure the prompts generated are detailed and simple for the text-to-image diffusion model."},
            {"role": "user", "content": f"Generate a general prompt and a series of detailed prompts from the following story: {story_text}. Ensure the prompts are suitable for a text-to-image diffusion model. Include a small description next to any mentioned person or animal to help the diffusion model understand the characters well. Split the general prompt and detailed prompts into two sections."}
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

# Main function
def main():
    base_dir = '/home/rjayanth/StoryDiffusion/backblaze_prompts'  # Updated path for the example
    output_dir = '/home/rjayanth/StoryDiffusion/generated_prompts'
    os.makedirs(output_dir, exist_ok=True)

    # Process each volume
    for volume_number in range(20):  # Adjust the range as needed
        story_file_path = f"{base_dir}/output_story_combined_volume_{volume_number}.txt"
        output_file_path = f"{output_dir}/generated_prompts_volume_{volume_number}.txt"

        print(f"Processing volume {volume_number}...")

        # Read the story
        story_text = read_story(story_file_path)

        # Generate prompts
        prompts = generate_prompts(story_text)

        # Save prompts to a text file
        save_prompts(prompts, output_file_path)

        print(f"Prompts for volume {volume_number} saved to {output_file_path}")

if __name__ == "__main__":
    main()
