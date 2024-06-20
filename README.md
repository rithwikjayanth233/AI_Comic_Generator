**Dream Tails: AI-Driven Comic Generation**

Welcome to the official GitHub repository for the Dream Tails project, as well as innovative AI initiatives under the Mechanical and AI Lab at Carnegie Mellon University. 
These projects leverage cutting-edge machine learning techniques to transform textual descriptions into dynamic visual and audio media, fostering new ways to interact with digital content.


**Dream Tails**

Dream Tails focuses on generating animated visual narratives from structured text inputs. 
Using advanced diffusion models, this project aims to create consistent and sequential image outputs that maintain object and character consistency across frames.
This work supports a variety of applications, from educational tools to entertainment and beyond, pushing the boundaries of automated content creation.


**Getting Started**

### 1. Clone the repository
First, clone the repository to your local machine using the following command:
```
git clone https://github.com/rithwikjayanth233/AI_Comic_Generator.git
cd AI_Comic_Generator
```

### 2. Create a virtual environment
Create a Python virtual environment to manage dependencies separately from your system Python. You can do this by running:
```
python -m venv venv
acitvate venv
```

### 3. Download requirements.txt
```
pip install -r requirements.txt

```

### 4. Obtain a GPT-4 Subscription
Ensure you have access to OpenAI's GPT-4 model. Subscribe and obtain an API key from [OpenAI's platform](https://platform.openai.com/).


### 5. Run the script to generate new prompts
Generate new comic prompts from the storyline by running the following script:
```
python backblaze_2_final_prompts.py
```

### 6. Delete previous outputs
To remove previously generated images and PDFs from Backblaze, execute:
```
python delete_files_backblaze.py
```

### 7. Generate the comic
Finally, generate the comic based on the newly created prompts by running:
```
python comic_generation_new.py
```


### Contributions

We welcome contributions from the community, whether they are bug fixes, improvements, or new features. Please refer to our contribution guidelines before making a pull request.


### License

This project is licensed under the terms of the MIT license.


### Acknowledgments

This repository for Dream Tails utilizes methods and insights developed by the StoryDiffusion initiative, particularly in advancing our text-to-image diffusion technologies. We acknowledge and appreciate the foundational work provided by StoryDiffusion, which has been crucial in the development of our innovative content creation tools within the Dream Tails project.

For further details on StoryDiffusion and their contributions to the field, please visit their official website:
StoryDiffusion

We are also thankful to all the open-source communities and contributors whose tools and libraries have been instrumental in facilitating our project's progress.



### Contact

For more information, questions, or partnership inquiries, please contact the project maintainers at rjayanth@andrew.cmu.edu.

Thank you for visiting our repository, and we look forward to seeing how our work can be applied and extended within the AI and creative communities!
