import pandas as pd
import ast
from collections import Counter

file_path = "/content/drive/MyDrive/dataset/full_dataset.csv"
df = pd.read_csv(file_path)

df["ingredients"] = df["ingredients"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)


ingredient_counter = Counter()
for ingredients in df["ingredients"]:
    ingredient_counter.update(ingredients)


print("🔹 Top 50 most common ingredients:")
print(ingredient_counter.most_common(50))

print(f"✅ Dataset contains {len(df)} recipes with {df['title'].nunique()} unique dish names.")


ingredient_counter = Counter()
for ingredients in df["ingredients"]:
    ingredient_counter.update(ingredients)

def filter_recipe(ingredients):
    common_count = sum(1 for ing in ingredients if ing in ingredient_counter)
    return 2 <= common_count <= 15

df_filtered = df[df["ingredients"].apply(filter_recipe)]

df_sampled = df_filtered.sample(n=50000, random_state=42)

df_sampled.to_csv("filtered_recipenlg_50k.csv", index=False)

print(f"✅ Dataset reduced to {len(df_sampled)} rows and saved as filtered_recipenlg_50k.csv")



import pandas as pd
import requests
import numpy as np
import time
import json
import os

file_path = "/content/filtered_recipenlg_50k.csv"
df = pd.read_csv(file_path)

JINA_API_URL = "https://api.jina.ai/v1/embeddings"
HEADERS = {
    "Content-Type": "application/json",
    "Authorization": "Bearer **"
}

df['recipe_text'] = df['title'] + " " + df['ingredients'] + " " + df['directions']

EMBEDDINGS_FILE = "recipe_embeddings.npy"
IDS_FILE = "recipe_ids.npy"
LOG_FILE = "processed_batches.txt"

if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(IDS_FILE):
    embeddings_list = list(np.load(EMBEDDINGS_FILE).tolist())
    recipe_ids = list(np.load(IDS_FILE).tolist())
    processed_batches = set(map(int, open(LOG_FILE).read().split())) if os.path.exists(LOG_FILE) else set()
else:
    embeddings_list = []
    recipe_ids = []
    processed_batches = set()


def get_embeddings(text_list):
    data = {
        "model": "jina-clip-v2",
        "dimensions": 1024,
        "normalized": True,
        "embedding_type": "float",
        "input": [{"text": text} for text in text_list]
    }

    response = requests.post(JINA_API_URL, headers=HEADERS, json=data)

    if response.status_code == 200:
        return json.loads(response.text)["data"]
    else:
        print(f"❌ Error: {response.status_code} - {response.text}")
        return None


BATCH_SIZE = 100
MAX_BATCHES = 50

for i in range(0, BATCH_SIZE * MAX_BATCHES, BATCH_SIZE):
    batch_num = i // BATCH_SIZE + 1

    if batch_num > MAX_BATCHES:
        break

    if batch_num in processed_batches:
        print(f"✅ Skipping batch {batch_num}, already processed.")
        continue

    batch_texts = df['recipe_text'][i:i+BATCH_SIZE].tolist()
    batch_ids = df['id'][i:i+BATCH_SIZE].tolist()

    print(f"🔹 Processing batch {batch_num}...")

    batch_embeddings = get_embeddings(batch_texts)

    if batch_embeddings:
        embeddings_list.extend(batch_embeddings)
        recipe_ids.extend(batch_ids)

        np.save(EMBEDDINGS_FILE, np.array(embeddings_list))
        np.save(IDS_FILE, np.array(recipe_ids))

        with open(LOG_FILE, "a") as f:
            f.write(f"{batch_num}\n")

        print(f"✅ Batch {batch_num} saved successfully!")

    time.sleep(2)

print("✅ Embedding process complete for 50 batches!")
