import pandas as pd
import json
import re
import csv
import os
import ast
from .utils import AzureOpenAIClient

# ---------- Triple Extraction Module ----------
def extract_triples(text: str) -> list:
    """Extracts triples from a given text using ast.literal_eval and regex as a fallback."""
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
def save_response(subject, triples, output_file):
    """Saves the generated triples for a subject into a CSV file. 
    If the file is empty, it writes headers as the first line.
    Otherwise, it appends the new data."""
    header = [subject, triples]
    file_exists = os.path.exists(output_file)
    
    with open(output_file, 'a', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_ALL)
        
        if not file_exists or os.stat(output_file).st_size == 0:
            writer.writerow(header)  # Write column headers if the file doesn't exist or is empty

        writer.writerow([subject, json.dumps(triples)])  # Append new row with data


# ---------- Correction Prompt Generation Module ----------
def generate_correction_prompt(base_prompt, subject, original_triples, errors):
    """Generates a correction prompt including errors and original triples."""
    correction_prompt = f"""{base_prompt}

    The following triples for the subject '{subject}' need attention:

    Errors:
    {errors}

    Original Triples:
    {json.dumps(original_triples, indent=2)}

    Please provide a corrected set of triples for this subject, addressing the errors mentioned above and considering whether the unused declarations are necessary or can be removed.
    """
    return correction_prompt

# ---------- Triple Generation Main Module ----------
def generate_triples(subjects_file, properties_file, output_file, ground_truth_file, correction_file=None, sample_size=30):
    with open(properties_file) as f:
        properties_data = json.load(f)
    
    with open(subjects_file) as f:
        subjects_data = json.load(f)

    # Load ground truth sentences
    ground_truth = pd.read_csv(ground_truth_file)

    subjects_dict = {}
    for item in subjects_data['data']:
        subjects_dict.update(item)

    if correction_file:
        existing_responses = pd.read_csv(output_file)
        correction_prompts = pd.read_csv(correction_file)
        sample_subjects = existing_responses.sample(n=sample_size) if sample_size else existing_responses
    else:
        sample_subjects = pd.DataFrame(list(subjects_dict.keys()), columns=['subject'])
        if sample_size:
            sample_subjects = sample_subjects.sample(n=sample_size)

    for index, row in sample_subjects.iterrows():
        subject = row['subject']
        # Fetch the ground truth sentence
        ground_truth_sentence = ground_truth[ground_truth['subject'] == subject]['combined_sentence'].values[0]

        original_triples = extract_triples(row['triples']) if 'triples' in row else []
        if correction_file:
            error_data = correction_prompts[correction_prompts['subject'] == subject]
            errors = error_data['correction_prompt'].values[0] if not error_data.empty else None
        else:
            errors = None

        subject_data = subjects_dict.get(subject, None)
        if subject_data:
            properties = build_properties_constraints(subject_data, properties_data)
            base_prompt = f"""Given the sentence "{ground_truth_sentence},

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

            The list should begin with all "instance of" triples, followed by the other relationship triples.

            Provide no additional information beyond the list of triples.
            """
            
            if errors:
                prompt = generate_correction_prompt(base_prompt, subject, original_triples, errors)
            else:
                prompt = f"{base_prompt}\n\nGenerate triples for the sentence:"

            print(prompt)
            
            # Call OpenAI API to generate or correct triples
            client = AzureOpenAIClient()
            client._validate_env_vars()
            client._configure_openai_api()
            response = client.send_request("", prompt)  
            response = response.choices[0].message.content

            print(response)

            new_triples = extract_triples(response)
            if new_triples:
                save_response(subject, new_triples, output_file)
                print(f"Saved {'corrected' if errors else 'new'} response for subject {subject}")
            else:
                print(f"Warning: No valid triples found for subject {subject}")
        else:
            print(f"Subject {subject} not found in JSON data.")

    print("Processing completed.")