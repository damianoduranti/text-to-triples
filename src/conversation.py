import pandas as pd
import json
import re
import csv
import os
import ast
from .utils import AzureOpenAIClient
from src.triple_analysis import analyze_triples

# ---------- Triple Extraction Module ----------
def extract_triples(text: str) -> list:
    """Extracts triples from a given text using ast.literal_eval and regex as a fallback."""
    if not text or text.strip() == "":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list) and all(isinstance(item, list) and len(item) == 3 for item in parsed):
            return parsed
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing response with ast.literal_eval: {e}")

    try:
        triple_pattern = re.compile(r'\[\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*,\s*["\']([^"\']+)["\']\s*\]')
        matches = triple_pattern.findall(text)
        triples = [list(match) for match in matches]
        if triples:
            return triples
    except Exception as e:
        print(f"Error extracting triples with regex: {e}")

    return []

# ---------- Property Constraints Builder Module ----------
def build_properties_constraints(subject_data, properties_data):
    """Builds a textual representation of property constraints based on subject data and property data."""
    ids = [item['property_id'] for item in subject_data if item['data_type'] != "Url"]
    labels = [item['label'] for item in subject_data if item['data_type'] != "Url"]
    
    text = ""
    for n in range(len(ids)):
        prop_data = properties_data.get(ids[n], {"subject_type_constraints": ["None"], "value_type_constraints": ["None"]})
        subject_type_constraints = ', '.join(prop_data.get('subject_type_constraints', ['None']))
        value_type_constraints = ', '.join(prop_data.get('value_type_constraints', ['None']))
        
        if subject_type_constraints and not value_type_constraints:
            text += f"{labels[n]} (Subject type: {subject_type_constraints}), "
        elif value_type_constraints and not subject_type_constraints:
            text += f"{labels[n]} (Value type: {value_type_constraints}), "
        elif subject_type_constraints and value_type_constraints:
            text += f"{labels[n]} (Subject type: {subject_type_constraints}, Value type: {value_type_constraints}), "
        else:
            text += f"{labels[n]}, "
    
    return text

# ---------- Response Saving Module ----------
def save_response(subject, triples, output_file, cycle):
    """Saves or updates the generated triples for a subject into a CSV file."""
    try:
        # Read existing responses
        responses_df = pd.read_csv(output_file, quoting=csv.QUOTE_MINIMAL)
    except FileNotFoundError:
        # If file doesn't exist, create a new DataFrame
        responses_df = pd.DataFrame(columns=['subject', 'triples', 'cycle'])

    # Convert triples to JSON string for correct escaping
    triples_str = json.dumps(triples)

    # Check if subject already exists
    if subject in responses_df['subject'].values:
        # Update existing row
        responses_df.loc[responses_df['subject'] == subject, 'triples'] = triples_str
        responses_df.loc[responses_df['subject'] == subject, 'cycle'] = cycle
    else:
        # Append new row
        new_row = pd.DataFrame({'subject': [subject], 'triples': [triples_str], 'cycle': [cycle]})
        responses_df = pd.concat([responses_df, new_row], ignore_index=True)

    # Save the updated DataFrame
    responses_df.to_csv(output_file, index=False)

# ---------- Correction Prompt Generation Module ----------
def generate_correction_prompt(base_prompt, original_triples, errors):
    """Generates a correction prompt including errors and original triples."""
    correction_prompt = f"""{base_prompt}

Original Triples:
{json.dumps(original_triples, indent=2) if original_triples else "No original triples found."}

Errors:
{errors}

Remember to follow all the guidelines mentioned in the base prompt, including using only the specified properties and ensuring all relationships are consistent with the ontology.
"""
    return correction_prompt

# ---------- Triple Generation Main Module ----------
def generate_triples(subjects_file, properties_file, output_file, ground_truth_file, chunk_output_dir, correction_file=None, cycle=0):
    with open(properties_file) as f:
        properties_data = json.load(f)
    
    with open(subjects_file) as f:
        subjects_data = json.load(f)

    if cycle > 1:
        previous_cycle_file = os.path.join(chunk_output_dir, f'responses_cycle_{cycle-1}.csv')
        if os.path.exists(previous_cycle_file):
            previous_responses = pd.read_csv(previous_cycle_file)
            previous_subjects = set(previous_responses['subject'])
        else:
            previous_subjects = set()

    ground_truth = pd.read_csv(ground_truth_file)

    try:
        existing_responses = pd.read_csv(output_file)
        processed_subjects = set(existing_responses['subject'])
    except FileNotFoundError:
        existing_responses = pd.DataFrame(columns=['subject', 'triples', 'cycle'])
        processed_subjects = set()

    subjects_dict = {}
    for item in subjects_data['data']:
        subjects_dict.update(item)

    print(f"Correction file: {correction_file}")

    if correction_file:
        correction_prompts = pd.read_csv(correction_file)
        all_subjects = pd.DataFrame(correction_prompts['subject'].unique(), columns=['subject'])
    else:
        all_subjects = pd.DataFrame(list(subjects_dict.keys()), columns=['subject'])

    print(f"Processing {len(subjects_dict)} subjects in cycle {cycle}")

    if correction_file and os.path.exists(correction_file):
        with open(correction_file, 'r') as f:
            correction_content = f.read().strip()
            if correction_content == "subject,correction_prompt":
                print(f"Correction file for cycle {cycle} is empty. Ending iteration.")
                return None

    total_subjects = len(all_subjects)
    processed_count = 0

    for index, row in all_subjects.iterrows():
        subject = row['subject']

        # Check if the subject has already been processed in this cycle
        if subject in processed_subjects:
            print(f"Skipping subject {subject} as it has already been processed in this cycle.")
            processed_count += 1
            continue
        
        # Fetch the ground truth sentence
        ground_truth_sentence = ground_truth[ground_truth['subject'] == subject]['combined_sentence'].values[0]

        original_triples = None
        if cycle > 1:
            previous_cycle_file = os.path.join(chunk_output_dir, f'responses_cycle_{cycle-1}.csv')
            if os.path.exists(previous_cycle_file):
                previous_responses = pd.read_csv(previous_cycle_file)
                previous_row = previous_responses[previous_responses['subject'] == subject]
                if not previous_row.empty:
                    original_triples = json.loads(previous_row.iloc[0]['triples'])

        if correction_file:
            error_data = correction_prompts[correction_prompts['subject'] == subject]
            errors = error_data['correction_prompt'].values[0] if not error_data.empty else None
        else:
            errors = None

        subject_data = subjects_dict.get(subject, None)
        if subject_data:
            properties = build_properties_constraints(subject_data, properties_data)
            base_prompt = f"""Given the sentence "{ground_truth_sentence}",

            Follow these steps:
            1. Identify instances of classes mentioned in the information.
            2. Create "instance of" triples for each identified instance.
            3. Use only the properties explicitly mentioned in the information.
            4. Ensure that the relationships you list are directly stated or can be unambiguously inferred from the information.
            5. Use only the following exact properties: {properties}
            6. Apply properties only to instances that comply with the ontology constraints.
            7. Validate that all relationships are consistent with the ontology.

            The response must be a list of complete triples with brackets in the form ['subject', 'property', 'object'] where:
            - 'subject' is an instance of a class from the ontology
            - 'property' is either "instance of" or one of the exact specified properties
            - 'object' is either a class from the ontology (for "instance of"), another instance, or a literal value

            Each triple must comply with the ontology constraints.
            Use only information from the sentence.

            The list should begin with all "instance of" triples, followed by the other relationship triples.

            Provide no additional information beyond the list of triples.
            """

            if errors:
                prompt = generate_correction_prompt(base_prompt, original_triples, errors)
            else:
                prompt = f"{base_prompt}\n\nGenerate triples for the sentence:"
            
            # Call OpenAI API to generate or correct triples
            client = AzureOpenAIClient()
            client._validate_env_vars()
            client._configure_openai_api()
            response = client.send_request("", prompt)  
            response = response.choices[0].message.content

            print(prompt)
            print(response)

            triples = extract_triples(response)
            save_response(subject, triples, output_file, cycle)
            print(f"Saved new response for subject {subject}")

        else:
            print(f"Subject {subject} not found in JSON data.")

        processed_subjects.add(subject)
        processed_count += 1
        completion_percentage = (processed_count / total_subjects) * 100
        print(f"Processing completed for {subject}: {completion_percentage:.2f}% done.")

    if cycle > 1:
        current_responses = pd.read_csv(output_file)
        current_subjects = set(current_responses['subject'])
        
        previous_cycle_file = os.path.join(chunk_output_dir, f'responses_cycle_{cycle-1}.csv')
        if os.path.exists(previous_cycle_file):
            previous_responses = pd.read_csv(previous_cycle_file)
            previous_subjects = set(previous_responses['subject'])
            
            missing_subjects = previous_subjects - current_subjects

            if missing_subjects:
                print(f"Found {len(missing_subjects)} responses from the previous cycle not in the current cycle. Adding them now.")
                
                for subject in missing_subjects:
                    previous_row = previous_responses[previous_responses['subject'] == subject].iloc[0]
                    new_row = pd.DataFrame({'subject': [subject], 'triples': [previous_row['triples']], 'cycle': [cycle]})
                    current_responses = pd.concat([current_responses, new_row], ignore_index=True)

                current_responses.to_csv(output_file, index=False)
                print(f"Updated {output_file} with responses from the previous cycle.")

    print("Recalculating metrics after adding missing subjects...")
    results = analyze_triples(output_file, ground_truth_file, properties_file, subjects_file)
    
    print("Processing completed.")
    return results