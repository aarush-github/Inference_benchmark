import json
import os
import random
import urllib.request

RAW_DIR = "data/sharegpt/raw"
PROCESSED_DIR = "data/sharegpt/processed"
RAW_FILE = os.path.join(RAW_DIR, "ShareGPT_V3_unfiltered_cleaned_split.json")
URL = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
SEED = 42
SPLITS = {1000: "sharegpt_1000.jsonl", 5000: "sharegpt_5000.jsonl", 10000: "sharegpt_10000.jsonl"}

def prepare_datasets():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    if not os.path.exists(RAW_FILE):
        print(f"Downloading raw dataset to {RAW_FILE}...")
        urllib.request.urlretrieve(URL, RAW_FILE)

    with open(RAW_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Extract first-turn human prompts
    prompts = []
    for item in data:
        conversations = item.get("conversations", [])
        if not conversations or conversations[0].get("from") != "human":
            continue
        
        prompt = conversations[0].get("value", "").strip()
        if len(prompt) > 10:
            prompts.append({"id": item.get("id"), "prompt": prompt})

    # Deterministic shuffle for even token distribution
    random.Random(SEED).shuffle(prompts)

    # Generate nested, overlapping splits
    for count, filename in SPLITS.items():
        subset = prompts[:count]
        output_path = os.path.join(PROCESSED_DIR, filename)
        
        with open(output_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(subset):
                record = {"id": item["id"] or f"sharegpt_{i:06d}", "prompt": item["prompt"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                
        print(f"Created {filename}: {len(subset):,} prompts")

if __name__ == "__main__":
    prepare_datasets()