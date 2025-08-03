# Recipe Generator with Step-by-Step Visuals

This project is a web application that generates culinary recipes based on provided ingredients. It creates a step-by-step guide with generated images for each cooking step, and can also produce a video of the entire cooking process.

## Features

- **Generate Recipes**: Input ingredients to generate unique recipes.
- **Step-by-Step Images**: Visualize each cooking step with AI-generated images.
- **Video Generation**: Create detailed cooking videos using AI models.

## Components

- **app.py**: Main application file that integrates all components and provides the Streamlit interface.
- **Data Files**:
  - `filtered_recipenlg_50k.csv`: Dataset containing recipes.
  - `recipe_embeddings.npy` & `recipe_ids.npy`: Precomputed embeddings for recipe similarity.
- **Auxiliary Scripts**: Additional Python files for specific tasks.

## How to Use

1. **Enter Ingredients**: Provide ingredients in a comma-separated format.
2. **Generate Recipe**: Click the 'Generate Recipe' button to create a new recipe.
3. **Navigate Steps**: Use the step navigation buttons to browse through cooking instructions and generated images.
4. **Create Video**: Optionally, generate a cooking video by clicking the 'Generate Video' button.

## Setup and Installation

1. **Install Required Libraries**:
    - diffusers, transformers, torch, torchvision, pandas, numpy, opencv-python, moviepy, etc.
2. **Run the Application**:
    - Use Streamlit to run the application: `streamlit run app.py`

## Technologies Used

- **Streamlit**: For the user interface.
- **Modal**: API and processing management.
- **FAISS**: Efficient similarity searching for recipe recommendations.
- **Groq and Stability AI**: AI models for recipe generation and image/video rendering.
- **Stable Diffusion 3**: For generating images for each recipe step.
- **Wan2.1-T2V-1.3B**: For advanced video generation capabilities.

## Requirements

Make sure you have the necessary API keys and environment setup to connect to AI models and data sources.

## Licenses and Credits

This project uses external AI models and data that are subject to their respective licenses and terms of use. Please make sure to comply with them.
