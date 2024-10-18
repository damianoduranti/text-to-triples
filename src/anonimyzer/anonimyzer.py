import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON, POST
import time
import json
import ast
import re
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_new_entities_by_classes(classes, exclude_entities_same_class, total_limit):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setMethod(POST)

    exclude_filter = ""
    if exclude_entities_same_class:
        exclude_filter = "FILTER(?item NOT IN (" + ", ".join(f"wd:{eid}" for eid in exclude_entities_same_class) + ")) ."

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

    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    try:
        results = sparql.query().convert()
        return [(result["item"]["value"].split('/')[-1], result["itemLabel"]["value"]) 
                for result in results["results"]["bindings"]]
    except Exception as e:
        print(f"Error in SPARQL query: {e}")
        return []

def batch_sparql_query(entity_ids, query_template, batch_size=50):
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    results = {}
    entity_ids = list(entity_ids)
    total_batches = (len(entity_ids) + batch_size - 1) // batch_size
    for i in range(0, len(entity_ids), batch_size):
        batch = entity_ids[i:i+batch_size]
        current_batch = i // batch_size + 1
        logging.info(f"Processing batch {current_batch}/{total_batches}")
        entities_str = " ".join(f"wd:{eid}" for eid in batch)
        query = query_template.format(entities_str)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        try:
            query_results = sparql.query().convert()
            for result in query_results["results"]["bindings"]:
                entity_id = result["entity"]["value"].split('/')[-1]
                process_result(results, entity_id, result)
        except Exception as e:
            logging.error(f"Error in batch query: {e}")
    return results

def process_result(results, entity_id, result):
    if "class" in result:
        class_id = result["class"]["value"].split('/')[-1]
        class_label = result["classLabel"]["value"]
        results.setdefault(entity_id, []).append((class_id, class_label))
    elif "entityLabel" in result:
        results[entity_id] = result["entityLabel"]["value"]

def get_wikidata_classes_batch(entity_ids):
    query_template = """
    SELECT ?entity ?class ?classLabel WHERE {{
      VALUES ?entity {{ {} }}
      ?entity p:P31/ps:P31 ?class .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    return batch_sparql_query(entity_ids, query_template)

def get_entity_labels(entity_ids):
    query_template = """
    SELECT ?entity ?entityLabel WHERE {{
      VALUES ?entity {{ {} }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
    }}
    """
    return batch_sparql_query(entity_ids, query_template)

def get_subject_object_classes_and_build_pools(df):
    start_time = time.time()
    logging.info(f"Starting to process {len(df)} rows")
    
    all_subject_ids = set(df['subject'].unique())
    all_object_ids = set()

    logging.info(f"Parsing subject and object matchings")
    df['subject_matching'] = df['subject_matching'].apply(ast.literal_eval)
    df['object_matching'] = df['object_matching'].apply(ast.literal_eval)

    logging.info("Extracting all subject and object IDs")
    for matching in df['subject_matching']:
        all_subject_ids.update([entry[0] for entry in matching])
    for matching in df['object_matching']:
        all_object_ids.update([entry[0] for entry in matching])

    logging.info(f"Found {len(all_subject_ids)} unique subject IDs and {len(all_object_ids)} unique object IDs")

    logging.info("Getting Wikidata classes for subjects")
    subject_classes = get_wikidata_classes_batch(all_subject_ids)
    logging.info("Getting Wikidata classes for objects")
    object_classes = get_wikidata_classes_batch(all_object_ids)

    logging.info("Building subject pool")
    subject_pool = build_entity_pool(subject_classes)
    logging.info("Building object pool")
    object_pool = build_entity_pool(object_classes)

    logging.info(f"Total processing time: {time.time() - start_time:.2f} seconds")
    return subject_pool, object_pool, subject_classes, object_classes

def build_entity_pool(entity_classes):
    class_to_ids = {}
    for entity_id, classes in entity_classes.items():
        class_ids = tuple(sorted([class_id for class_id, _ in classes]))
        class_to_ids.setdefault(class_ids, set()).add(entity_id)

    entity_pool = {}
    for class_ids, entity_ids in class_to_ids.items():
        n_substitutions_needed = len(entity_ids)
        entity_pool[class_ids] = get_new_entities_by_classes(
            list(class_ids), entity_ids, total_limit=n_substitutions_needed)

    return entity_pool

def replace_entities_in_sentence(sentence, new_matching, original_matching):
    replacements = []
    for i, (new_entry, original_entry) in enumerate(zip(new_matching, original_matching)):
        if original_entry is None:
            logging.warning(f"Skipping None original_entry at index {i}")
            continue
        if len(original_entry) < 2:
            logging.warning(f"Skipping short original_entry at index {i}: {original_entry}")
            continue
        new_label = new_entry[1]
        original_synonyms = set(original_entry[1:])
        sorted_synonyms = sorted(original_synonyms, key=len, reverse=True)
        for synonym in sorted_synonyms:
            replacements.append((synonym, new_label))
    
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    
    for original, replacement in replacements:
        pattern = r'\b{}\b'.format(re.escape(original))
        sentence = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
    
    return sentence

def main():
    logging.info("Starting main process")
    df = pd.read_csv('data/ground_truth.csv')
    logging.info(f"Loaded {len(df)} rows from CSV")

    subject_pool, object_pool, subject_classes, object_classes = get_subject_object_classes_and_build_pools(df)

    logging.info("Saving pools")
    save_pools(subject_pool, object_pool)
    logging.info("Loading pools")
    subject_pool, object_pool = load_pools()

    logging.info("Creating replacements")
    subject_replacements = create_replacements(subject_classes, subject_pool)
    object_replacements = create_replacements(object_classes, object_pool)

    logging.info("Applying replacements to subjects")
    df['new_subject'] = df['subject'].map(lambda x: subject_replacements.get(x, (x,))[0])
    df['original_subject_matching'] = df['subject_matching'].copy()
    df['subject_matching'] = df['subject_matching'].apply(lambda x: replace_entities_in_matching(x, subject_replacements))

    logging.info("Applying replacements to objects")
    df['original_object_matching'] = df['object_matching'].copy()
    df['object_matching'] = df['object_matching'].apply(lambda x: replace_entities_in_matching(x, object_replacements))

    logging.info("Replacing entities in sentences")
    for index, row in df.iterrows():
        try:
            df.at[index, 'combined_sentence'] = replace_entities_in_sentence(
                row['combined_sentence'], 
                row['subject_matching'],
                row['original_subject_matching']
            )
        except Exception as e:
            logging.error(f"Error processing subject for row {index}: {e}")
            logging.error(f"Row data: {row.to_dict()}")

    for index, row in df.iterrows():
        try:
            df.at[index, 'combined_sentence'] = replace_entities_in_sentence(
                row['combined_sentence'], 
                row['object_matching'],
                row['original_object_matching']
            )
        except Exception as e:
            logging.error(f"Error processing object for row {index}: {e}")
            logging.error(f"Row data: {row.to_dict()}")

    logging.info("Saving anonymized data")
    columns_to_keep = ['subject', 'combined_sentence', 'subject_matching', 'predicate_matching', 'object_matching']
    df[columns_to_keep].to_csv('data/ground_truth_anonymized.csv', index=False)
    logging.info("Anonymized data has been saved to 'data/ground_truth_anonymized.csv'")

def save_pools(subject_pool, object_pool):
    with open('subject_pool.json', 'w') as f:
        json.dump({str(k): v for k, v in subject_pool.items()}, f, indent=4)
    with open('object_pool.json', 'w') as f:
        json.dump({str(k): v for k, v in object_pool.items()}, f, indent=4)

def load_pools():
    with open('subject_pool.json', 'r') as f:
        subject_pool = {eval(k): v for k, v in json.load(f).items()}
    with open('object_pool.json', 'r') as f:
        object_pool = {eval(k): v for k, v in json.load(f).items()}
    return subject_pool, object_pool

def create_replacements(entity_classes, entity_pool):
    replacements = {}
    for entity_id, classes in entity_classes.items():
        class_ids = tuple(sorted([class_id for class_id, _ in classes]))
        if class_ids in entity_pool and entity_pool[class_ids]:
            replacements[entity_id] = entity_pool[class_ids].pop()
    return replacements

def replace_entities_in_matching(matching_list, replacements):
    return [
        (replacements.get(entry[0], (entry[0], entry[1]))[0], replacements.get(entry[0], (entry[0], entry[1]))[1])
        for entry in matching_list
    ]

if __name__ == "__main__":
    main()
