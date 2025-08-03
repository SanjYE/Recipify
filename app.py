import streamlit as st
import faiss
import numpy as np
import pandas as pd
import requests
import json
import re
import time
import hashlib
import os
import shutil
import subprocess
import traceback
from datetime import datetime
import uuid
import threading
import glob
from groq import Groq
from gradio_client import Client
import modal


if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'recipe' not in st.session_state:
    st.session_state.recipe = None
if 'steps' not in st.session_state:
    st.session_state.steps = []
if 'image_paths' not in st.session_state:
    st.session_state.image_paths = {}
if 'ingredients' not in st.session_state:
    st.session_state.ingredients = ""
if 'generating_recipe' not in st.session_state:
    st.session_state.generating_recipe = False
if 'generating_images' not in st.session_state:
    st.session_state.generating_images = False
if 'recipe_history' not in st.session_state:
    st.session_state.recipe_history = {}


if 'generate_more_clicked' not in st.session_state:
    st.session_state.generate_more_clicked = False
if 'need_rerun' not in st.session_state:
    st.session_state.need_rerun = False
if 'recipe_counter' not in st.session_state:
    st.session_state.recipe_counter = 0
if 'previous_title' not in st.session_state:
    st.session_state.previous_title = ""
if 'video_requested' not in st.session_state:
    st.session_state.video_requested = False
if 'video_processing' not in st.session_state:
    st.session_state.video_processing = False
if 'video_ready' not in st.session_state:
    st.session_state.video_ready = False
if 'video_path' not in st.session_state:
    st.session_state.video_path = ""


@st.cache_resource
def load_api_clients():
    groq_client = Groq(api_key="**")
    sd_client = Client("stabilityai/stable-diffusion-3-medium")
    return groq_client, sd_client

groq_client, sd_client = load_api_clients()


cuda_version = "12.8.0"  
flavor = "devel"  
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

volume = modal.Volume.from_name("genai-results", create_if_missing=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .apt_install("git", "libgl1-mesa-glx", "libglib2.0-0", "ffmpeg")
    .pip_install(
        "diffusers",
        "transformers",
        "torch",
        "torchvision",
        "datasets",
        "wandb",
        "bitsandbytes",
        "peft",
        "sentencepiece",
        "git+https://github.com/huggingface/diffusers.git",
        "ftfy",
        "opencv-python",
        "imageio",
        "imageio-ffmpeg"
    )
)

app = modal.App("recipe_video_generation", image=image)


@app.function(gpu="A100-80GB", timeout=3600, volumes={"/results": volume})
def generate_recipe_video_function(recipe_prompt, output_filename):
    import torch
    from diffusers.utils import export_to_video
    from diffusers import AutoencoderKLWan, WanPipeline
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
    import os

    model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
    flow_shift = 5.0 
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=flow_shift)
    pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
    pipe.scheduler = scheduler
    pipe.to("cuda")

    negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

    output = pipe(
        prompt=recipe_prompt,
        negative_prompt=negative_prompt,
        height=720,
        width=1280,
        num_frames=81,
        guidance_scale=5.0,
    ).frames[0]
    
    output_path = f"/results/{output_filename}"
    export_to_video(output, output_path, fps=16)
    
    if not os.path.exists(output_path):
        raise FileNotFoundError(f"Video file {output_path} was not created")
        
    return output_filename

def generate_recipe_video(prompt, filename="output.mp4"):
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        remote_filename = f"recipe_video_{timestamp}.mp4"
        
        st.session_state.modal_debug = {
            "timestamp": timestamp,
            "remote_filename": remote_filename,
            "prompt_length": len(prompt),
            "start_time": datetime.now().strftime("%H:%M:%S")
        }
        
        try:
            st.session_state.modal_debug["modal_start_time"] = datetime.now().strftime("%H:%M:%S")
            with app.run():
                result = generate_recipe_video_function.remote(prompt, remote_filename)
            st.session_state.modal_debug["modal_end_time"] = datetime.now().strftime("%H:%M:%S")
            st.session_state.modal_debug["result"] = result
        except Exception as e:
            error_msg = f"Error in Modal function call: {str(e)}"
            st.error(error_msg)
            st.session_state.modal_debug["modal_error"] = error_msg
            return None
            
        return remote_filename
        
    except Exception as e:
        error_msg = f"Error generating video: {str(e)}"
        st.error(error_msg)
        st.session_state.modal_debug = st.session_state.get("modal_debug", {})
        st.session_state.modal_debug["error"] = error_msg
        return None

def execute_video_generation():
    try:
        if not st.session_state.get('process_video_on_next_run', False):
            return
 
        st.session_state.process_video_on_next_run = False

        recipe_directions = st.session_state.recipe_for_video
        timestamp = st.session_state.video_timestamp
  
        prompt = f"A detailed cooking video showing the following recipe being prepared step by step in a modern kitchen: {recipe_directions}"
        
        with st.spinner("Starting video generation on Modal... This may take several minutes."):
            remote_filename = generate_recipe_video(prompt)
            
            if remote_filename:
                local_path = os.path.join(videos_dir, f"recipe_video_{timestamp}.mp4")
                
                with st.spinner("Video generated! Downloading..."):
                    if check_and_download_video(remote_filename, local_path):
                        st.session_state.video_ready = True
                        st.session_state.video_path = local_path
                        st.success("Video downloaded successfully! Expand the Recipe Video section to view it.")
                    else:
                        st.session_state.video_ready = False
                        st.session_state.remote_filename = remote_filename  # Save for later retrieval attempts
                        st.warning("Video was created but couldn't be downloaded automatically. Use the 'Check Status' button to try again.")
            else:
                st.error("Failed to generate video. Please try again.")
                
        st.session_state.video_processing = False
        
    except Exception as e:
        st.session_state.video_processing = False
        st.error(f"Error in video generation: {str(e)}")
        st.session_state.modal_debug = st.session_state.get("modal_debug", {})
        st.session_state.modal_debug["error_trace"] = traceback.format_exc()

def download_video_from_modal(remote_filename, local_path):
    try:
        os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)

        command = f"modal volume get genai-results {remote_filename} {local_path} --force"

        result = subprocess.run(
            command, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        if result.returncode == 0 and os.path.exists(local_path):
            st.session_state.video_ready = True
            st.session_state.video_path = local_path
            st.session_state.video_processing = False
            return True
        else:
            return False
            
    except Exception as e:
        st.error(f"Error downloading video: {str(e)}")
        return False

def check_and_download_video(remote_filename, local_path, max_attempts=3):
    for attempt in range(max_attempts):
        try:
            if download_video_from_modal(remote_filename, local_path):
                return True
            time.sleep(5)  
        except Exception as e:
            st.warning(f"Attempt {attempt+1} failed: {str(e)}")
            time.sleep(5)
    
    return False

def start_video_generation(recipe_directions):
    st.session_state.video_processing = True
    st.session_state.video_requested = True
    st.session_state.video_ready = False

    st.session_state.recipe_for_video = recipe_directions

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.video_timestamp = timestamp
    
    st.session_state.process_video_on_next_run = True
    
    st.rerun()
    
    return "Video generation started in the background. This may take several minutes."


if 'process_video_on_next_run' in st.session_state and st.session_state.process_video_on_next_run:
    execute_video_generation()

def check_video_status():
    if 'remote_filename' in st.session_state and not st.session_state.video_ready:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        local_path = os.path.join(videos_dir, f"recipe_video_check_{timestamp}.mp4")
        
        with st.spinner("Checking if video is ready..."):
            if check_and_download_video(st.session_state.remote_filename, local_path):
                st.session_state.video_ready = True
                st.session_state.video_path = local_path
                st.success("Video is now ready to view!")
                return "Video is now ready to view!"
            else:
                st.warning("Video is still processing. Please check again later.")
                return "Video is still processing. Please check again later."
    elif st.session_state.video_ready:
        return "Video is ready to view!"
    else:
        return "No video has been requested or an error occurred."

# Load data
@st.cache_resource
def load_data():
    df = pd.read_csv("C:/Users/sanja/Desktop/GENAI/filtered_recipenlg_50k.csv")
    embeddings = np.load("C:/Users/sanja/Desktop/GENAI/recipe_embeddings.npy", allow_pickle=True)
    recipe_ids = np.load("C:/Users/sanja/Desktop/GENAI/recipe_ids.npy", allow_pickle=True)

    df = df[df["id"].isin(recipe_ids)]

    sorted_indices = np.argsort(recipe_ids)
    recipe_ids = recipe_ids[sorted_indices]
    embeddings = embeddings[sorted_indices]
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return df, embeddings, recipe_ids, index

df, embeddings, recipe_ids, index = load_data()


images_dir = "images"
os.makedirs(images_dir, exist_ok=True)

videos_dir = "videos"
os.makedirs(videos_dir, exist_ok=True)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "**"
}

# Function to get query embedding
def get_query_embedding(text):
    data = {
        "model": "jina-clip-v2",
        "dimensions": 1024,
        "normalized": True,
        "embedding_type": "float",
        "input": [{"text": text}]
    }

    response = requests.post(JINA_API_URL, headers=HEADERS, json=data)

    if response.status_code == 200:
        response_data = json.loads(response.text)
        embedding_vector = np.array(response_data["data"][0]["embedding"], dtype=np.float32)
        return embedding_vector
    else:
        st.error(f"Error: {response.status_code} - {response.text}")
        return None


def retrieve_similar_recipes(user_query, k=5):
    query_embedding = get_query_embedding(user_query)

    if query_embedding is None:
        st.error("Error generating query embedding. Try again.")
        return None

    query_embedding = query_embedding.reshape(1, -1)
    distances, indices = index.search(query_embedding, k)

    matched_ids = [recipe_ids[i] for i in indices[0] if i < len(recipe_ids)]
    results = df[df["id"].isin(matched_ids)]
    
    return results[["title", "ingredients", "directions"]]


def generate_recipe(user_query, retrieved_recipes, feedback=None):
    
    query_hash = hashlib.md5(user_query.lower().strip().encode()).hexdigest()
    history_count = st.session_state.recipe_history.get(query_hash, 0)
    st.session_state.recipe_history[query_hash] = history_count + 1
    
    temperature = min(0.7 + (history_count * 0.15), 1.3)  
    
    if 'generated_titles' not in st.session_state:
        st.session_state.generated_titles = {}
    
    
    prev_titles = st.session_state.generated_titles.get(query_hash, [])
    
  
    st.session_state.prev_titles_debug = prev_titles.copy()
    
    retrieved_text = "\n\n".join(
        f"Title: {row['title']}\nIngredients: {row['ingredients']}\nDirections: {row['directions']}"
        for _, row in retrieved_recipes.iterrows()
    )


    feedback_text = ""
    if feedback:
        feedback_text = f"""
        IMPORTANT - Previous generation attempt had these issues:
        {feedback}
        
        Please address these specific issues in your new recipe and create a VALID food combination.
        Focus on creating harmonious flavor profiles and logical ingredient combinations.
        Consider completely changing the approach if necessary - don't just make minor adjustments.
        """
    

    variation_text = ""
    if history_count > 0:
        prev_titles_str = "\\n".join(prev_titles) if prev_titles else "None yet"
        
        variation_text = f"""
        IMPORTANT: This is request #{history_count+1} for these same ingredients. 
        Please generate a COMPLETELY DIFFERENT recipe than before.
        
        Previously generated titles for these ingredients were:
        {prev_titles_str}
        
        YOU MUST CREATE A TOTALLY DIFFERENT RECIPE with a DIFFERENT TITLE than any listed above.
        
        Be creative and explore different cooking styles, cuisines, or preparation methods.
        Consider:
        - Different cooking methods (baking, frying, steaming, etc.)
        - Different cuisine inspirations (Italian, Asian, Mexican, etc.)
        - Different textures and presentations
        - Different flavor profiles (spicy, sweet, savory, etc.)
        """

    prompt = f"""
    The user wants a recipe with the following ingredients: {user_query}.

    Here are some similar recipes:
    {retrieved_text}
    
    {feedback_text}
    
    {variation_text}

    Based on these, generate a **new recipe** that:
    - Uses the user-provided ingredients in a LOGICAL and HARMONIOUS way
    - Creates VALID flavor combinations that would actually taste good
    - Follows a structured format: **Title, Ingredients, Directions**
    - Ensures the recipe makes culinary sense (appropriate cooking techniques, temperatures, etc.)
    - Make sure each step in the directions is on a separate line and is clear
    - The last line should be the "name of recipe"
    
    IMPORTANT: Focus on creating a recipe that will pass a validity check for logical food combinations!
    """

    seed = int(time.time()) % 10000 + (history_count * 1000) + np.random.randint(0, 10000)
    
    st.session_state.seed_debug = seed

    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert chef AI."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=0.95,
        seed=seed,
        stream=False,
    )

    return completion.choices[0].message.content

def perform_validity_check(generated_recipe):
    new_prompt = f"""
    This is the generated recipe: {generated_recipe}.

    Perform a detailed validity check on the generated recipe to determine if the food combination is valid or not.
    Evaluate against these specific metrics for food combinations:
    
    1. Flavor Profile Compatibility
    - Are there compatible flavor compounds between ingredients?
    - Do the flavors work well together or clash?
    
    2. Taste Balance
    - Is there a good balance of the five basic tastes: sweet, salty, sour, bitter, and umami?
    - Does any single taste overwhelm or conflict with others?
    
    3. Texture Harmony
    - Do the textures complement or contrast pleasantly?
    - Are there any conflicting textures that create an unpleasant mouthfeel?
    
    4. Cultural & Contextual Expectations
    - Does the combination make sense within some culinary tradition?
    - Is the combination logical in the context of how we normally eat food?
    
    5. Temperature & Serving Logic
    - Do the hot/cold elements work well together?
    - Is the serving method appropriate for the ingredients?
    
    6. Ingredient Function
    - Are ingredients being used in appropriate ways (bases, condiments, highlight flavors)?
    - Is there a clear main ingredient and supporting elements?
    
    IMPORTANT: Start your response with either "VALID:" or "INVALID:" followed by your detailed assessment.
    If invalid, clearly explain which aspects need to be fixed and suggest specific improvements.
    If valid, explain what makes this a successful recipe.
    
    Be decisive and direct in your evaluation.
    """
    
    completion = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are an expert chef AI and food critic with deep knowledge of culinary science and flavor combinations."},
            {"role": "user", "content": new_prompt}
        ],
        temperature=0.7,
        top_p=1,
        stream=False,
    )
    
    return completion.choices[0].message.content


def is_recipe_valid(validity_result):
    if validity_result.strip().upper().startswith("VALID:"):
        return True
    if validity_result.strip().upper().startswith("INVALID:"):
        return False
    
    lower_result = validity_result.lower()

    invalid_indicators = [
        "invalid", "not valid", "illogical", "doesn't work", "does not work", 
        "poor combination", "bad combination", "incompatible", "clash", "disjointed",
        "lacks balance", "disrupts the harmony", "unconventional", "may not appeal"
    ]
    
    valid_indicators = [
        "valid", "well-balanced", "harmonious", "complementary", "works well",
        "good combination", "flavorful", "tasty", "delicious", "balanced"
    ]
    
    for indicator in invalid_indicators:
        if indicator in lower_result:
            return False
    
    for indicator in valid_indicators:
        if indicator in lower_result:
            return True

    positive_count = sum(1 for word in ["good", "nice", "great", "excellent", "perfect", "balanced", "complementary"] 
                         if word in lower_result)
    negative_count = sum(1 for word in ["not", "doesn't", "don't", "clash", "odd", "unusual", "strange"] 
                         if word in lower_result)
    

    return positive_count > negative_count


def parse_recipe_sections(recipe_text):
    
    title_pattern = r'(?:\*\*Title:\*\*|^[*#]+\s*Title:\s*|\*\*|#\s*)(.*?)(?=\*\*Ingredients|\n\s*\*\*Ingredients|\n\s*###\s*Ingredients|Ingredients:|\Z)'
    ingredients_pattern = r'(?:\*\*Ingredients:\*\*|Ingredients:|###\s*Ingredients:)(.*?)(?=\*\*Directions|Directions:|###\s*Directions:|\Z)'
    directions_pattern = r'(?:\*\*Directions:\*\*|Directions:|###\s*Directions:)(.*)'
    
    title_match = re.search(title_pattern, recipe_text, re.DOTALL | re.IGNORECASE)
    ingredients_match = re.search(ingredients_pattern, recipe_text, re.DOTALL | re.IGNORECASE)
    directions_match = re.search(directions_pattern, recipe_text, re.DOTALL | re.IGNORECASE)
    
    title = title_match.group(1).strip() if title_match else ""
    ingredients = ingredients_match.group(1).strip() if ingredients_match else ""
    directions = directions_match.group(1).strip() if directions_match else ""
    
    st.session_state.debug_info = {
        "raw_recipe": recipe_text[:500], 
        "title_match": title_match.group(0) if title_match else "No match",
        "extracted_title": title
    }
    

    if not title or len(title) < 2:
        first_line = recipe_text.strip().split('\n')[0].strip()
        if not first_line.lower().startswith(('ingredients', 'directions')):
            title = first_line
    

    if not title or len(title) < 2:
        lines = recipe_text.split('\n')
        for line in lines[-3:]:  
            if "name of recipe" in line.lower():
                title = line.split(":")[-1].strip() if ":" in line else line.replace("name of recipe", "").strip()
                break
    

    if not title or len(title) < 2:
        title = f"Recipe with {st.session_state.ingredients.split(',')[0]}"
    
    
    if directions:
        lines = directions.split('\n')
        clean_lines = [line.strip() for line in lines if line.strip()]
        if len(clean_lines) > 1 and ("name of recipe" in clean_lines[-1].lower() or not clean_lines[-1].startswith(('1', '-'))):
            directions = '\n'.join(clean_lines[:-1])
        else:
            directions = '\n'.join(clean_lines)
    

    steps = [step.strip() for step in directions.split('\n') if step.strip()]
    
    return {
        "title": title,
        "ingredients": ingredients,
        "directions": directions,
        "steps": steps
    }


def generate_image_for_step(step_text, step_index):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]
    filename = f"step_{step_index}_{timestamp}_{unique_id}.webp"
    saved_image_path = os.path.join(images_dir, filename)
    
    try:
        result = sd_client.predict(
            prompt=step_text,
            negative_prompt="text, watermark, signature, blurry, distorted, low quality",
            seed=0,
            randomize_seed=True,
            width=1024,
            height=768,
            guidance_scale=7,
            num_inference_steps=28,
            api_name="/infer"
        )

        temp_image_path = result[0]
   
        shutil.copy2(temp_image_path, saved_image_path)
        
        return saved_image_path
    except Exception as e:
        st.error(f"Error generating image: {e}")
        return None


def generate_valid_recipe(ingredients):
    with st.spinner("🔍 Retrieving similar recipes..."):
        retrieved_recipes = retrieve_similar_recipes(ingredients, k=5)
        
    if retrieved_recipes is None:
        st.error("No recipes retrieved. Try different ingredients.")
        return None
    
    with st.spinner("👨‍🍳 Generating recipe..."):
        generated_recipe = generate_recipe(ingredients, retrieved_recipes)
    
    with st.spinner("🧐 Checking recipe validity..."):
        validity_result = perform_validity_check(generated_recipe)
        is_valid = is_recipe_valid(validity_result)
    
    max_attempts = 5
    attempt_count = 1
    
    while not is_valid and attempt_count < max_attempts:
        with st.spinner(f"🔄 Recipe invalid, regenerating (Attempt {attempt_count+1}/{max_attempts})..."):
            specific_feedback = f"""
            {validity_result}
            
            To fix this recipe, you need to:
            1. Reconsider flavor compatibility between ingredients
            2. Ensure taste balance (sweet, salty, sour, bitter, umami)
            3. Create logical texture combinations
            4. Respect cultural and contextual food expectations
            5. Consider completely changing the approach with these ingredients
            """
      
            regenerated_recipe = generate_recipe(ingredients, retrieved_recipes, feedback=specific_feedback)
         
            generated_recipe = regenerated_recipe
          
            validity_result = perform_validity_check(generated_recipe)
            is_valid = is_recipe_valid(validity_result)
            
            attempt_count += 1
    
    if not is_valid:
        st.error("Sorry, we couldn't generate a valid recipe after multiple attempts. Please try with different ingredients.")
        return None
    
    try:
        recipe_sections = parse_recipe_sections(generated_recipe)
        return recipe_sections
    except Exception as e:
        st.error(f"Error parsing the recipe: {e}")
        return None


def get_image_filename(recipe_id, step_index):
    return os.path.join(images_dir, f"recipe_{recipe_id}_step_{step_index}.webp")


def image_exists_for_step(recipe_id, step_index):
    filename = get_image_filename(recipe_id, step_index)
    return os.path.exists(filename)


def generate_recipe_id(ingredients, title):
    combined = f"{ingredients}_{title}".lower().replace(" ", "_")
    return hashlib.md5(combined.encode()).hexdigest()[:10]


def next_step():
    if st.session_state.current_step < len(st.session_state.steps) - 1:
        st.session_state.current_step += 1


def prev_step():
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1


def generate_more():
    st.session_state.generate_more_clicked = True


st.title("🍳 Recipe Generator with Step-by-Step Images")


with st.sidebar:
    st.header("About")
    st.write("""
    This app generates recipes based on your ingredients and creates images for each step.
    
    How to use:
    1. Enter ingredients in the text box
    2. Click 'Generate Recipe'
    3. Navigate through the recipe steps with the buttons
    4. Use 'More Recipes' to generate alternative recipes with the same ingredients
    5. Click 'Generate Video' to create a cooking video with AI
    """)
    
    st.header("Image Generation")
    st.write("""
    Images are generated using Stable Diffusion 3.
    This may take some time, especially for recipes with many steps.
    """)
    
    if st.session_state.generating_images:
        st.warning("⏳ Generating images in background... Images will appear as they are ready.")
        
    st.header("Video Generation")
    st.write("""
    Videos are generated using Wan2.1-T2V-1.3B model running on Modal.
    This process typically takes 5-15 minutes as it runs on a remote GPU.
    """)
    
    if st.session_state.video_processing:
        st.warning("⏳ Generating video in background... This may take several minutes.")
    elif st.session_state.video_ready:
        st.success("✅ Video is ready to view!")
    
    if 'debug_info' in st.session_state and st.session_state.debug_info:
        with st.expander("Debug Info", expanded=False):
            st.write("Recipe Parsing Debug:")
            st.write(f"Title extracted: '{st.session_state.debug_info['extracted_title']}'")
            st.write(f"Title pattern match: '{st.session_state.debug_info['title_match']}'")
            st.write("Raw recipe start:")
            st.code(st.session_state.debug_info['raw_recipe'], language="markdown")
            
            if 'recipe_debug' in st.session_state:
                st.write("---")
                st.write("Recipe Structure Debug:")
                st.write(f"Has title key: {st.session_state.recipe_debug['has_title']}")
                st.write(f"Title value: '{st.session_state.recipe_debug['title_value']}'")
                st.write(f"Available keys: {', '.join(st.session_state.recipe_debug['keys'])}")
                st.write(f"Steps count: {st.session_state.recipe_debug['steps_count']}")
                
                if 'history_count' in st.session_state.recipe_debug:
                    st.write(f"History count: {st.session_state.recipe_debug['history_count']}")
                if 'prev_titles' in st.session_state.recipe_debug:
                    st.write("Previously generated titles:")
                    for i, title in enumerate(st.session_state.recipe_debug['prev_titles']):
                        st.write(f"{i+1}. {title}")
                
            if 'recipe' in st.session_state and st.session_state.recipe:
                st.write("---")
                st.write("Current Recipe State:")
                if 'title' in st.session_state.recipe:
                    st.write(f"Current title: '{st.session_state.recipe['title']}'")
                else:
                    st.write("Current recipe has no title key!")
                    
          
            if 'generated_titles' in st.session_state and st.session_state.ingredients:
                st.write("---")
                st.write("Title Generation Debug:")
                query_hash = hashlib.md5(st.session_state.ingredients.lower().strip().encode()).hexdigest()
                if query_hash in st.session_state.generated_titles:
                    st.write(f"All titles for current ingredients:")
                    for i, title in enumerate(st.session_state.generated_titles[query_hash]):
                        st.write(f"{i+1}. {title}")
                else:
                    st.write("No titles stored for current ingredients.")
                    
                if 'seed_debug' in st.session_state:
                    st.write(f"Last seed used: {st.session_state.seed_debug}")
                if 'recipe_history' in st.session_state and query_hash in st.session_state.recipe_history:
                    st.write(f"Current history count: {st.session_state.recipe_history[query_hash]}")
                    
            
            st.write("---")
            st.write("Video Generation Debug:")
            st.write(f"Video requested: {st.session_state.get('video_requested', False)}")
            st.write(f"Video processing: {st.session_state.get('video_processing', False)}")
            st.write(f"Video ready: {st.session_state.get('video_ready', False)}")
            if 'video_path' in st.session_state and st.session_state.video_path:
                st.write(f"Video path: {st.session_state.video_path}")
                st.write(f"File exists: {os.path.exists(st.session_state.video_path)}")
            if 'remote_filename' in st.session_state:
                st.write(f"Remote filename: {st.session_state.remote_filename}")
            if 'process_video_on_next_run' in st.session_state:
                st.write(f"Process on next run: {st.session_state.process_video_on_next_run}")
            if 'recipe_for_video' in st.session_state:
                st.write("Recipe directions for video (first 100 chars):")
                st.code(st.session_state.recipe_for_video[:100] + "...")
                
            if 'modal_debug' in st.session_state:
                st.write("---")
                st.write("Modal Debug Information:")
                for key, value in st.session_state.modal_debug.items():
                    st.write(f"{key}: {value}")


with st.container():
    col1, col2 = st.columns([3, 1])
    
    with col1:
        new_ingredients = st.text_input("Enter ingredients (comma separated):", 
                                     placeholder="e.g., chicken, rice, onions, garlic",
                                     key="ingredient_input")
    
    with col2:
        generate_button = st.button("Generate Recipe", type="primary", use_container_width=True)
        
    if generate_button and new_ingredients:
        st.session_state.ingredients = new_ingredients
        
        st.session_state.current_step = 0
       
        query_hash = hashlib.md5(new_ingredients.lower().strip().encode()).hexdigest()
        if query_hash in st.session_state.recipe_history:
            st.session_state.recipe_history[query_hash] = 0
     
        if 'generated_titles' not in st.session_state:
            st.session_state.generated_titles = {}
        st.session_state.generated_titles[query_hash] = []

        st.session_state.video_requested = False
        st.session_state.video_processing = False
        st.session_state.video_ready = False
        st.session_state.video_path = ""
        if 'remote_filename' in st.session_state:
            del st.session_state.remote_filename
        
        st.session_state.generating_recipe = True
      
        new_recipe = generate_valid_recipe(new_ingredients)
        
        if new_recipe:
            st.session_state.recipe_debug = {
                'has_title': 'title' in new_recipe,
                'title_value': new_recipe.get('title', 'NO TITLE'),
                'keys': list(new_recipe.keys()),
                'steps_count': len(new_recipe.get('steps', [])),
                'is_initial': True
            }
            
            if not new_recipe.get('title') or len(new_recipe.get('title', '')) < 2:
                # Provide a fallback title
                new_recipe['title'] = f"Recipe with {new_ingredients.split(',')[0].strip()}"
                
            
            if 'title' in new_recipe and new_recipe['title']:
                if query_hash not in st.session_state.generated_titles:
                    st.session_state.generated_titles[query_hash] = []
                st.session_state.generated_titles[query_hash].append(new_recipe['title'])
        
        st.session_state.recipe = new_recipe
        
        if st.session_state.recipe:
            st.session_state.steps = st.session_state.recipe["steps"]
            st.session_state.recipe_id = generate_recipe_id(
                st.session_state.ingredients, 
                st.session_state.recipe["title"]
            )
            
         
            st.session_state.image_paths = {}
            
          
            if len(st.session_state.steps) > 0:
                with st.spinner("Generating first step image..."):
                    step_text = st.session_state.steps[0]
                    filename = get_image_filename(st.session_state.recipe_id, 0)
                    
                    # Only generate if it doesn't exist
                    if not os.path.exists(filename):
                        image_path = generate_image_for_step(step_text, 0)
                        if image_path and os.path.exists(image_path):
                            try:
                                shutil.copy2(image_path, filename)
                                st.session_state.image_paths[0] = filename
                            except Exception as e:
                                st.error(f"Error saving image: {e}")
                    else:
                        st.session_state.image_paths[0] = filename
        
        st.session_state.generating_recipe = False
        
        # Force a rerun to update the UI
        st.rerun()

# Check if we need to generate a new recipe based on the flag
if st.session_state.generate_more_clicked and st.session_state.ingredients:
    # Reset the flag
    st.session_state.generate_more_clicked = False
    
    # Store previous title for comparison
    if st.session_state.recipe and 'title' in st.session_state.recipe:
        st.session_state.previous_title = st.session_state.recipe['title']
        
        # Add current title to list of generated titles to avoid regenerating the same recipe
        query_hash = hashlib.md5(st.session_state.ingredients.lower().strip().encode()).hexdigest()
        if 'generated_titles' not in st.session_state:
            st.session_state.generated_titles = {}
        if query_hash not in st.session_state.generated_titles:
            st.session_state.generated_titles[query_hash] = []
        
        # Add the current title to the list if it's not already there
        current_title = st.session_state.recipe['title']
        if current_title not in st.session_state.generated_titles[query_hash]:
            st.session_state.generated_titles[query_hash].append(current_title)
    
    # Reset current step
    st.session_state.current_step = 0
    
    # Set generating_recipe to true to show spinner
    st.session_state.generating_recipe = True
    
    # Generate a new recipe with the same ingredients
    new_recipe = generate_valid_recipe(st.session_state.ingredients)
    
    # Store debug information
    if new_recipe:
        st.session_state.recipe_debug = {
            'has_title': 'title' in new_recipe,
            'title_value': new_recipe.get('title', 'NO TITLE'),
            'keys': list(new_recipe.keys()),
            'steps_count': len(new_recipe.get('steps', [])),
            'history_count': st.session_state.recipe_history.get(query_hash, 0),
            'prev_titles': st.session_state.generated_titles.get(query_hash, [])
        }
        
        # Ensure the recipe has a title
        if not new_recipe.get('title') or len(new_recipe.get('title', '')) < 2:
            # Provide a fallback title
            new_recipe['title'] = f"Recipe with {st.session_state.ingredients.split(',')[0].strip()}"
    
    # Update session state with the new recipe
    st.session_state.recipe = new_recipe
    
    if st.session_state.recipe:
        # Increment recipe counter
        st.session_state.recipe_counter += 1
        
        # Update steps and generate recipe ID for image filenames
        st.session_state.steps = st.session_state.recipe["steps"]
        st.session_state.recipe_id = generate_recipe_id(
            st.session_state.ingredients, 
            st.session_state.recipe["title"]
        )
        
        # Clear image paths cache
        st.session_state.image_paths = {}
        
        # Generate first image in the main thread 
        # (but other images will be generated on-demand as user navigates)
        if len(st.session_state.steps) > 0:
            with st.spinner("Generating first step image..."):
                step_text = st.session_state.steps[0]
                filename = get_image_filename(st.session_state.recipe_id, 0)
                
                # Only generate if it doesn't exist
                if not os.path.exists(filename):
                    image_path = generate_image_for_step(step_text, 0)
                    if image_path and os.path.exists(image_path):
                        try:
                            shutil.copy2(image_path, filename)
                            st.session_state.image_paths[0] = filename
                        except Exception as e:
                            st.error(f"Error saving image: {e}")
                else:
                    st.session_state.image_paths[0] = filename
    
    st.session_state.generating_recipe = False
    
    # Force a rerun to update the UI
    st.rerun()

if st.session_state.recipe:
    st.markdown("---")
    
    col1, col2, col3 = st.columns([4, 1, 1])
    with col1:
        # Ensure we have a valid title
        if 'title' not in st.session_state.recipe or not st.session_state.recipe['title']:
            st.session_state.recipe['title'] = f"Recipe with {st.session_state.ingredients.split(',')[0].strip()}"
            
        st.header(f"🍽️ {st.session_state.recipe['title']}")
        
    
        if st.session_state.previous_title and st.session_state.previous_title != st.session_state.recipe['title']:
            st.success(f"New recipe generated! (Previous: {st.session_state.previous_title})")
    
    with col2:
        st.button("More Recipes", on_click=generate_more, type="secondary", use_container_width=True)
        if st.session_state.recipe_counter > 0:
            st.caption(f"Recipe variation #{st.session_state.recipe_counter+1}")
            
    with col3:
       
        if not st.session_state.video_requested:
            video_button = st.button("Generate Video", type="primary", use_container_width=True)
            if video_button:
    
                message = start_video_generation(st.session_state.recipe["directions"])
                st.info(message)
        elif st.session_state.video_processing:
            st.button("Generating...", disabled=True, use_container_width=True)
            st.spinner("Video is being generated. This may take several minutes.")
        else:
            
            check_button = st.button("Check Status", type="primary", use_container_width=True)
            if check_button:
                status = check_video_status()
                st.info(status)
    
   
    with st.expander("📋 Ingredients", expanded=True):
        st.markdown(st.session_state.recipe["ingredients"])
        
    
    if st.session_state.video_ready and os.path.exists(st.session_state.video_path):
        with st.expander("🎬 Recipe Video", expanded=True):
            st.video(st.session_state.video_path)
            st.success("Recipe video successfully generated!")
    
    st.markdown("---")
    if st.session_state.recipe and 'steps' in st.session_state and st.session_state.steps:
        
        if 'title' not in st.session_state.recipe or not st.session_state.recipe['title']:
            st.session_state.recipe['title'] = f"Recipe with {st.session_state.ingredients.split(',')[0].strip()}"
            
        st.subheader(f"🍲 {st.session_state.recipe['title']}")
        
        # Step counter
        st.write(f"Step {st.session_state.current_step + 1} of {len(st.session_state.steps)}")
        
        
        current_step = st.session_state.steps[st.session_state.current_step]
        st.markdown(f"### {current_step}")
        
        
        if hasattr(st.session_state, 'recipe_id'):
            recipe_id = st.session_state.recipe_id
            step_index = st.session_state.current_step
            image_filename = get_image_filename(recipe_id, step_index)

            if os.path.exists(image_filename):
                # Image exists, display it
                st.image(image_filename, use_column_width=True)
            else:
                with st.spinner(f"Generating image for step {step_index + 1}..."):
                    temp_image_path = generate_image_for_step(current_step, step_index)
                    if temp_image_path and os.path.exists(temp_image_path):
                        try:
                            # Copy to permanent location with recipe-specific filename
                            shutil.copy2(temp_image_path, image_filename)
                            st.image(image_filename, use_column_width=True)
                        except Exception as e:
                            st.error(f"Error saving image: {e}")
                            # Still try to show the temp image
                            if os.path.exists(temp_image_path):
                                st.image(temp_image_path, use_column_width=True)
                    else:
                        st.info("Could not generate image for this step. Please try again.")
        else:
            st.info("Recipe information is incomplete. Please regenerate the recipe.")
        
        
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.current_step > 0:
                st.button("⬅️ Previous Step", on_click=prev_step, use_container_width=True)
        with col2:
            if st.session_state.current_step < len(st.session_state.steps) - 1:
                st.button("Next Step ➡️", on_click=next_step, use_container_width=True)
            else:
                st.button("Finish 🎉", on_click=lambda: None, use_container_width=True, disabled=True)
elif st.session_state.generating_recipe:
    st.spinner("Generating recipe...")
else:
    st.info("Enter ingredients and click 'Generate Recipe' to start.") 