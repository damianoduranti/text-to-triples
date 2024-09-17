import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, POST
import time
from urllib.error import HTTPError
import json
import ast
import re

# Load the CSV data
df = pd.read_csv('data/ground_truth.csv')

# Define the function to get new entities by classes
def get_new_entities_by_classes(classes, exclude_entities_same_class, total_limit):
    """
    Queries Wikidata for new entities that are direct instances of specified classes.
    Ensures that excluded entities of the same class are not in the results.
    """
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setMethod(POST)  # Use POST to handle long queries

    exclude_entities_list = list(exclude_entities_same_class)
    exclude_filter = ""
    if exclude_entities_list:
        # The list now contains only entities of the same class, so it's shorter
        exclude_filter = "FILTER(?item NOT IN (" + ", ".join(f"wd:{eid}" for eid in exclude_entities_list) + ")) ."

    class_values = " ".join(f"wd:{cls}" for cls in classes)

    query = f"""
    SELECT DISTINCT ?item ?itemLabel WHERE {{
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
      {{
        SELECT DISTINCT ?item WHERE {{
          ?item wdt:P31 ?class .
          VALUES ?class {{ {class_values} }} .
          {exclude_filter}
        }}
        LIMIT {total_limit}
      }}
    }}
    """

    print(f"SPARQL Query:\n{query}")

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        bindings = results["results"]["bindings"]
        new_entities = []
        for result in bindings:
            entity_id = result["item"]["value"].split('/')[-1]
            entity_label = result["itemLabel"]["value"]
            new_entities.append((entity_id, entity_label))
    except Exception as e:
        print(f"Error in SPARQL query: {e}")
        new_entities = []

    return new_entities

# Function to batch fetch classes for multiple entities
def get_wikidata_classes_batch(entity_ids):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    entity_ids = list(entity_ids)
    entity_classes = {}
    batch_size = 50  # Limit batch size to avoid overly long queries
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i:i+batch_size]
        entities_str = " ".join(f"wd:{eid}" for eid in batch)
        query = f"""
        SELECT ?entity ?class ?classLabel WHERE {{
          VALUES ?entity {{ {entities_str} }}
          ?entity p:P31/ps:P31 ?class .
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                entity_id = result["entity"]["value"].split('/')[-1]
                class_id = result["class"]["value"].split('/')[-1]
                class_label = result["classLabel"]["value"]
                entity_classes.setdefault(entity_id, []).append((class_id, class_label))
        except Exception as e:
            print(f"Error fetching classes: {e}")
            continue  # Skip this batch on error
    return entity_classes

# Function to get subject and object classes and build pools
def get_subject_object_classes_and_build_pools(df, sample_size=10):
    start_time = time.time()
    subject_classes = {}
    object_classes = {}
    subject_pool = {}
    object_pool = {}

    # Limit the dataset to the first sample_size rows
    df_sample = df.head(sample_size)

    # Extract all subject IDs from the sampled CSV, including those in 'subject_matching'
    all_subject_ids = set(df_sample['subject'].unique())

    # Convert 'subject_matching' and 'object_matching' columns to lists
    df_sample['subject_matching'] = df_sample['subject_matching'].apply(ast.literal_eval)
    df_sample['object_matching'] = df_sample['object_matching'].apply(ast.literal_eval)

    # Include IDs from 'subject_matching'
    for matching in df_sample['subject_matching']:
        all_subject_ids.update([entry[0] for entry in matching])

    # Extract all object IDs from 'object_matching'
    all_object_ids = set()
    for matching in df_sample['object_matching']:
        all_object_ids.update([entry[0] for entry in matching])

    # Batch fetch classes for subjects
    print("Fetching classes for subjects...")
    subject_classes = get_wikidata_classes_batch(all_subject_ids)
    print(f"Classes for subjects: {subject_classes}")

    # Batch fetch classes for objects
    print("Fetching classes for objects...")
    object_classes = get_wikidata_classes_batch(all_object_ids)
    print(f"Classes for objects: {object_classes}")

    # Build mappings from classes to entities
    subject_class_to_ids = {}
    for subject_id, classes in subject_classes.items():
        class_ids = tuple(sorted([class_id for class_id, _ in classes]))
        subject_class_to_ids.setdefault(class_ids, set()).add(subject_id)

    object_class_to_ids = {}
    for object_id, classes in object_classes.items():
        class_ids = tuple(sorted([class_id for class_id, _ in classes]))
        object_class_to_ids.setdefault(class_ids, set()).add(object_id)

    # Build replacement pools based on the identified classes
    for class_ids, subject_ids in subject_class_to_ids.items():
        n_substitutions_needed = len(subject_ids)
        print(f"Building subject pool for class(es): {class_ids}, need {n_substitutions_needed} entities")

        # Exclude only the subject entities that are of the same class(es)
        exclude_subjects_same_class = subject_ids  # subject_ids is already the set of entities of this class combination

        # Get new entities of the same class(es) to build a pool
        subject_pool[class_ids] = get_new_entities_by_classes(
            list(class_ids), exclude_subjects_same_class, total_limit=n_substitutions_needed)
        print(f"Subject pool for class(es) {class_ids}: {subject_pool[class_ids]}")

    # Build object pools
    for class_ids, object_ids in object_class_to_ids.items():
        n_substitutions_needed = len(object_ids)
        print(f"Building object pool for class(es): {class_ids}, need {n_substitutions_needed} entities")

        # Exclude only the object entities that are of the same class(es)
        exclude_objects_same_class = object_ids  # object_ids is the set of entities of this class combination

        # Get new entities of the same class(es) to build a pool
        object_pool[class_ids] = get_new_entities_by_classes(
            list(class_ids), exclude_objects_same_class, total_limit=n_substitutions_needed)
        print(f"Object pool for class(es) {class_ids}: {object_pool[class_ids]}")

    print(f"Total time for get_subject_object_classes_and_build_pools: {time.time() - start_time:.2f} seconds")
    return subject_pool, object_pool, subject_classes, object_classes

# Run the function and get the classes
subject_pool, object_pool, subject_classes, object_classes = get_subject_object_classes_and_build_pools(df, sample_size=1000)

# Save the pools to JSON files
# Convert tuple keys to strings for object_pool and subject_pool
subject_pool_serializable = {str(k): v for k, v in subject_pool.items()}
object_pool_serializable = {str(k): v for k, v in object_pool.items()}

# Save subject pool
with open('subject_pool.json', 'w') as subject_file:
    json.dump(subject_pool_serializable, subject_file, indent=4)

# Save object pool to JSON with stringified tuple keys
with open('object_pool.json', 'w') as object_file:
    json.dump(object_pool_serializable, object_file, indent=4)

print("Subject and object pools have been saved with stringified keys.")

# Load the pools from JSON files
with open('subject_pool.json', 'r') as subject_file:
    subject_pool_serializable = json.load(subject_file)
    subject_pool = {eval(k): v for k, v in subject_pool_serializable.items()}

with open('object_pool.json', 'r') as object_file:
    object_pool_serializable = json.load(object_file)
    object_pool = {eval(k): v for k, v in object_pool_serializable.items()}

# Load and prepare your data
df = pd.read_csv('data/ground_truth.csv')

# Convert matching columns to lists
df['subject_matching'] = df['subject_matching'].apply(ast.literal_eval)
df['object_matching'] = df['object_matching'].apply(ast.literal_eval)

# Create mappings for entity replacements

# Map original subjects to their replacement entities
subject_replacements = {}
for subject_id, classes in subject_classes.items():
    class_ids = tuple(sorted([class_id for class_id, _ in classes]))
    if class_ids in subject_pool and subject_pool[class_ids]:
        replacement = subject_pool[class_ids].pop()
        subject_replacements[subject_id] = replacement

# Map original objects to their replacement entities
object_replacements = {}
for object_id, classes in object_classes.items():
    class_ids = tuple(sorted([class_id for class_id, _ in classes]))
    if class_ids in object_pool and object_pool[class_ids]:
        replacement = object_pool[class_ids].pop()
        object_replacements[object_id] = replacement

# Apply replacements to your DataFrame

# Replace the 'subject' column
df['new_subject'] = df['subject'].apply(lambda x: subject_replacements.get(x, (x, ))[0])

# Replace entities in 'subject_matching'
def replace_subjects_in_matching(matching_list):
    new_matching_list = []
    for entry in matching_list:
        original_entity_id = entry[0]
        if original_entity_id in subject_replacements:
            replacement_id, replacement_label = subject_replacements[original_entity_id]
            entry = list(entry)  # Convert tuple to list to modify
            entry[0] = replacement_id
            entry[1] = replacement_label
        new_matching_list.append(tuple(entry))
    return new_matching_list

df['subject_matching'] = df['subject_matching'].apply(replace_subjects_in_matching)

# Replace entities in 'object_matching'
def replace_objects_in_matching(matching_list):
    new_matching_list = []
    for entry in matching_list:
        original_entity_id = entry[0]
        if original_entity_id in object_replacements:
            replacement_id, replacement_label = object_replacements[original_entity_id]
            entry = list(entry)  # Convert tuple to list to modify
            entry[0] = replacement_id
            entry[1] = replacement_label
        new_matching_list.append(entry)
    return new_matching_list

df['object_matching'] = df['object_matching'].apply(replace_objects_in_matching)

# Update text fields (e.g., 'combined_sentence')

# Fetch labels for original and replacement entities
def get_entity_labels(entity_ids):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    entity_labels = {}
    batch_size = 50
    entity_ids = list(entity_ids)
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i:i+batch_size]
        entities_str = " ".join(f"wd:{eid}" for eid in batch)
        query = f"""
        SELECT ?entity ?entityLabel WHERE {{
          VALUES ?entity {{ {entities_str} }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
        """
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            results = sparql.query().convert()
            for result in results["results"]["bindings"]:
                entity_id = result["entity"]["value"].split('/')[-1]
                entity_label = result["entityLabel"]["value"]
                entity_labels[entity_id] = entity_label
        except Exception as e:
            print(f"Error fetching labels: {e}")
            continue
    return entity_labels

# Collect all original and replacement entity IDs
original_entity_ids = set(subject_replacements.keys()).union(object_replacements.keys())

replacement_entity_ids = set([v[0] for v in subject_replacements.values()]).union(
    [v[0] for v in object_replacements.values()])

# Fetch labels
print("Fetching labels for original entities...")
original_labels = get_entity_labels(original_entity_ids)
print("Fetching labels for replacement entities...")
replacement_labels = get_entity_labels(replacement_entity_ids)

# Map original labels to replacement labels
label_replacements = {}
for original_id, (replacement_id, _) in subject_replacements.items():
    original_label = original_labels.get(original_id)
    replacement_label = replacement_labels.get(replacement_id)
    if original_label and replacement_label:
        label_replacements[original_label] = replacement_label

for original_id, (replacement_id, _) in object_replacements.items():
    original_label = original_labels.get(original_id)
    replacement_label = replacement_labels.get(replacement_id)
    if original_label and replacement_label:
        label_replacements[original_label] = replacement_label

# Replace labels in 'combined_sentence'
def replace_entities_in_sentence(sentence, label_replacements):
    # Use regex to replace whole words only
    pattern = re.compile(r'\b(' + '|'.join(map(re.escape, label_replacements.keys())) + r')\b')
    return pattern.sub(lambda x: label_replacements[x.group()], sentence)

df['combined_sentence'] = df['combined_sentence'].apply(lambda x: replace_entities_in_sentence(x, label_replacements))

# Save the anonymized data
df.to_csv('data/ground_truth_anonymized.csv', index=False)

print("Anonymized data has been saved to 'data/ground_truth_anonymized.csv'.")
