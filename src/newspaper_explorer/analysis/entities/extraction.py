import pandas as pd
import re
from germalemma import GermaLemma
from unidecode import unidecode
from pathlib import Path
from gliner import GLiNER
import ast
import json
import torch
from tqdm import tqdm


txts_path = Path("/mnt/data/userfiles/westphal/hackathon/dump")  #

lemmatizer = GermaLemma()


def normalize_german_text(text):
    text = str(text)
    text = text.replace("ẞ", "SS").replace("ß", "ss")
    text = text.replace("ſs", "ss").replace("ſ", "s")
    text = unidecode(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r'[^a-zäöüß .,;:!?\'"-]', "", text)
    tokens = text.split()
    lemmas = [lemmatizer.find_lemma(token, "NOUN") for token in tokens]
    return " ".join(lemmas).strip()


# Iterate txt files
normalized_dataset = {}

# Get list of files first for progress bar
txt_files = list(txts_path.rglob("*.txt"))
print(f"Processing {len(txt_files)} text files...")

for file in tqdm(txt_files, desc="Normalizing texts"):
    with open(file, "r", encoding="utf-8") as f:
        text = f.read()
    normalized_text = normalize_german_text(text)

    normalized_dataset[file.stem] = normalized_text

# Convert to DataFrame
normalized_output = pd.DataFrame(
    list(normalized_dataset.items()), columns=["id", "normalized_text"]
)
# Save the output to a new CSV file
normalized_output.to_csv(
    "rawtextblocks_normalized.csv", index=False, encoding="utf-8", sep="\t"
)


# Filter for texts with at least 1000 characters
filtered_df = normalized_output[normalized_output["normalized_text"].str.len() >= 1000]

# # Randomly sample 100 rows
# sampled_df = filtered_df.sample(n=min(100, len(filtered_df)), random_state=42)

# Initialize GLiNER
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# Move model to GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Using device: {device}")

# Labels for entity prediction
labels = ["Person", "Organisation", "Ereignis", "Ort"]

# Create empty lists to store results
all_ids = []
all_entities = []

# Process texts in batches for better performance
batch_size = 32  # Adjust based on GPU memory (higher = faster but more memory)
texts_to_process = []
ids_to_process = []

print(f"Processing {len(filtered_df)} texts for entity recognition...")

# Prepare all texts and IDs
for idx, row in filtered_df.iterrows():
    texts_to_process.append(row["normalized_text"][:500])
    ids_to_process.append(row["id"])

# Process in batches with progress bar using TRUE batch inference
for i in tqdm(
    range(0, len(texts_to_process), batch_size), desc="Entity recognition (batched)"
):
    batch_texts = texts_to_process[i : i + batch_size]
    batch_ids = ids_to_process[i : i + batch_size]

    # Process entire batch at once (much faster!)
    batch_entities = model.batch_predict_entities(batch_texts, labels, threshold=0.5)

    # Store results - batch_entities is a list of lists
    for text_id, entities in zip(batch_ids, batch_entities):
        for entity in entities:
            all_ids.append(text_id)
            all_entities.append({"text": entity["text"], "label": entity["label"]})

# Create new dataframe with results
results_df = pd.DataFrame({"id": all_ids, "entity": all_entities})

# Save to CSV
results_df.to_csv("entities_results.csv", index=False)


def serialize_entities(df):
    # Initialize the result dictionary
    result = {}

    # Convert string representation of dictionaries to actual dictionaries
    df["entity"] = df["entity"].apply(ast.literal_eval)

    # Group by ID
    grouped = df.groupby("id")

    # Process each group
    for text_id, group in grouped:
        # Initialize empty lists for each category
        entity_dict = {"Person": [], "Ort": [], "Ereignis": [], "Organisation": []}

        # Process each entity in the group
        for _, row in group.iterrows():
            entity = row["entity"]
            label = entity["label"]
            text = entity["text"].lower()  # Convert to lowercase

            if label in entity_dict:
                entity_dict[label].append(text)

        # Add to result
        result[text_id] = entity_dict

    return result


# Serialize the data
serialized_data = serialize_entities(results_df)

# If you want to save the result to a JSON file
with open("serialized_entities.json", "w", encoding="utf-8") as f:
    json.dump(serialized_data, f, ensure_ascii=False, indent=4)
