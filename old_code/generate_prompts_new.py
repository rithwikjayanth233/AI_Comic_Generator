import os
import openai

# Load OpenAI API key from a text file
def load_api_key(file_path):
    with open(file_path, 'r') as file:
        return file.read().strip()

# Path to the file containing the OpenAI API key
api_key_file = '/path/to/gpt_key.txt'
api_key = load_api_key(api_key_file)

# Initialize OpenAI client with the loaded API key
openai.api_key = api_key

# Function to generate text prompts using ChatGPT API
def generate_prompts(story_text, num_prompts):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that generates text prompts for a text-to-image diffusion model. Ensure the prompts generated are detailed and simple for the text-to-image diffusion model."},
            {"role": "user", "content": f"Generate a general prompt and {num_prompts} detailed prompts from the following story: {story_text}. Ensure the prompts are suitable for a text-to-image diffusion model. Include a small description next to any mentioned person or animal to help the diffusion model understand the characters well. Split the general prompt and detailed prompts into two sections. Make sure the general and detailed prompts start with a capital letter."}
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

# Function to process and combine general and detailed prompts
def process_prompts(prompts):
    lines = prompts.split("\n")
    general_prompt = ""
    character_descriptions = []
    detailed_prompts = []
    reading_detailed_prompts = False
    reading_character_descriptions = False

    for line in lines:
        if line.startswith("General Prompt:"):
            general_prompt = line[len("General Prompt:"):].strip()
        elif line.startswith("Character Descriptions:"):
            reading_character_descriptions = True
            reading_detailed_prompts = False
        elif line.startswith("Detailed Prompts:"):
            reading_detailed_prompts = True
            reading_character_descriptions = False
        elif reading_character_descriptions:
            if line.strip() and line.startswith("-"):
                character_descriptions.append(line.strip())
        elif reading_detailed_prompts:
            if line.strip():
                prompt = line.split('.', 1)[1].strip()
                detailed_prompts.append(prompt)

    # Combine character descriptions into the detailed prompts
    for i in range(len(detailed_prompts)):
        for description in character_descriptions:
            name = description.split(":")[0].strip()
            if name in detailed_prompts[i]:
                detailed_prompts[i] = f"{description.split(':')[0]} ({description.split(':')[1].strip()}), {detailed_prompts[i]}"

    # Capitalize the first letter of the general prompt and detailed prompts
    general_prompt = general_prompt.capitalize()
    detailed_prompts = [prompt.capitalize() for prompt in detailed_prompts]

    return general_prompt, detailed_prompts

# Function to save the final combined prompts to a text file
def save_final_prompts(general_prompt, detailed_prompts, output_path):
    with open(output_path, 'w') as file:
        file.write(f"General Prompt:\n\"{general_prompt}\"\n\nDetailed Prompts:\n")
        for i, prompt in enumerate(detailed_prompts, 1):
            file.write(f"{i}. \"{prompt}\"\n")

# Main function
def main():
    base_dir = '/path/to/backblaze_prompts'  # Updated path for the example
    output_dir = '/path/to/generated_prompts_new'
    os.makedirs(output_dir, exist_ok=True)

    # Process each volume
    for volume_number in range(20):  # Adjust the range as needed
        story_file_path = f"{base_dir}/output_story_combined_volume_{volume_number}.txt"
        output_file_path = f"{output_dir}/generated_prompts_volume_{volume_number}.txt"
        final_output_file_path = f"{output_dir}/final_prompts_volume_{volume_number}.txt"

        print(f"Processing volume {volume_number}...")

        # Read the story
        story_text = read_story(story_file_path)
        story_sentences = story_text.split('.')
        num_prompts = len(story_sentences)

        # Generate prompts
        prompts = generate_prompts(story_text, num_prompts)

        # Save prompts to a text file
        save_prompts(prompts, output_file_path)

        # Process and combine general and detailed prompts
        general_prompt, detailed_prompts = process_prompts(prompts)

        # Ensure the number of detailed prompts matches the number of sentences
        while len(detailed_prompts) < num_prompts:
            additional_prompts = generate_prompts(story_text, num_prompts - len(detailed_prompts))
            _, new_detailed_prompts = process_prompts(additional_prompts)
            detailed_prompts.extend(new_detailed_prompts)

        # Save the final combined prompts to a text file
        save_final_prompts(general_prompt, detailed_prompts, final_output_file_path)

        print(f"Final prompts for volume {volume_number} saved to {final_output_file_path}")

if __name__ == "__main__":
    main()
