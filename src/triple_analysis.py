import json
import ast
import pandas as pd
import re

# ---------- Data Loading Module ----------
class DataLoader:
    @staticmethod
    def load_json(file_path):
        """Load JSON data from a given file path."""
        with open(file_path, 'r') as f:
            return json.load(f)

    @staticmethod
    def load_properties(file_path):
        """Load properties from a JSON file."""
        return DataLoader.load_json(file_path)

    @staticmethod
    def load_ground_truth(file_path):
        """Load ground truth data from a CSV file."""
        return pd.read_csv(file_path)

    @staticmethod
    def load_all_data():
        """Load all relevant data (properties, responses, ground truth, subject properties)."""
        property_constraints = DataLoader.load_json('data/properties.json')
        properties = DataLoader.load_properties('data/properties.json')
        responses = pd.read_csv('data/anon/responses.csv')
        ground_truth = DataLoader.load_ground_truth('data/anon/ground_truth_anonymized.csv')
        subjects_data = DataLoader.load_json('data/subjects.json')['data']
        subject_properties_dict = {subj: set(prop['label'] for prop in props) 
                                   for subject_dict in subjects_data 
                                   for subj, props in subject_dict.items()}
        return property_constraints, properties, responses, ground_truth, subject_properties_dict

# ---------- Triple Processing Module ----------
class TripleProcessor:
    @staticmethod
    def extract_triples(text: str) -> list:
        """Extract triples from a string using regex."""
        triple_pattern = re.compile(r"\['(.*?)',\s*'(.*?)',\s*'(.*?)'\]")
        matches = triple_pattern.findall(text)
        return [(subject, predicate, obj) for subject, predicate, obj in matches 
                if (subject, predicate, obj) != ('subject', 'property', 'object')]

    @staticmethod
    def parse_ground_truth_triples(row):
        """Parse ground truth triples from a given row of data."""
        subject_matching = ast.literal_eval(row['subject_matching'])
        predicate_matching = ast.literal_eval(row['predicate_matching'])
        object_matching = ast.literal_eval(row['object_matching'])
        return [(subject_matching, predicate_matching, object_matching)]

# ---------- Property Handling Module ----------
class PropertyHandler:
    @staticmethod
    def get_property_id(property_label, properties):
        """Retrieve the property ID corresponding to a given property label."""
        return next((prop_id for prop_id, prop_info in properties.items() 
                     if prop_info.get('label') == property_label), None)

    @staticmethod
    def get_property_label(property_id, properties):
        """Retrieve the property label corresponding to a given property ID."""
        prop_info = properties.get(property_id)
        return prop_info.get('label') if prop_info else None

# ---------- Validation Module ----------
class Validator:
    @staticmethod
    def is_valid_instance_type(property_label, instance_type, properties, property_constraints, is_subject=False):
        """Check if the instance type is valid for a given property label."""
        property_id = PropertyHandler.get_property_id(property_label, properties)
        if not property_id:
            return False, f"No property ID found for label: {property_label}"

        constraint_key = 'subject_type_constraints' if is_subject else 'value_type_constraints'
        valid_types = property_constraints.get(property_id, {}).get(constraint_key, [])

        if not valid_types:
            return True, "No constraints, valid by default"

        if instance_type in valid_types:
            return True, "Correct"
        else:
            return False, f"Instance type '{instance_type}' not in valid types {valid_types}"

    @staticmethod
    def is_correct_instance_of(triple, properties, triples, property_constraints):
        """Validate whether a given 'instance of' triple is correct."""
        subject, _, instance_type = triple
        instance_of_triples = {t[0]: t[2] for t in triples if t[1] == 'instance of'}

        if subject not in instance_of_triples:
            return False, f"Subject '{subject}' is not defined as an instance."

        subject_used = any(t[0] == subject or t[2] == subject for t in triples if t[1] != "instance of")

        if not subject_used:
            return "unused", f"Subject '{subject}' is not used in other triples."

        for t in triples:
            if t[1] != "instance of":
                property_id = PropertyHandler.get_property_id(t[1], properties)
                if not property_id:
                    continue

                if t[0] == subject or t[2] == subject:
                    is_subject = t[0] == subject
                    is_valid, log = Validator.is_valid_instance_type(t[1], instance_type, properties, property_constraints, is_subject=is_subject)
                    if is_valid:
                        return True, "Correct"
                    else:
                        subject_or_object = "subject" if is_subject else "object"
                        return False, f"Instance type '{instance_type}' not valid for {subject_or_object} '{subject}' with property '{t[1]}'"

        return False, f"Instance type '{instance_type}' not associated with any property for subject '{subject}'"

    @staticmethod
    def is_correct_relation(triple, instances, properties, property_constraints, subject_properties):
        """Validate whether a given relation triple is correct."""
        subject, property_label, obj = triple
        property_id = PropertyHandler.get_property_id(property_label, properties)

        if not property_id:
            return False, f"Error: Property '{property_label}' not found in properties."

        if property_label not in subject_properties:
            return False, f"Relation '{property_label}' not present in subject properties for '{subject}'"

        subject_instance_type = instances.get(subject)
        object_instance_type = instances.get(obj)

        subject_error = object_error = None

        if not subject_instance_type:
            subject_error = f"Subject '{subject}' not defined as an instance."

        if not object_instance_type:
            object_error = f"Object '{obj}' not defined as an instance."

        # Return appropriate error based on which is undefined
        if subject_error and object_error:
            return False, f"{subject_error} and {object_error}"
        elif subject_error:
            return False, subject_error
        elif object_error:
            return False, object_error

        # If both are defined, proceed with further checks (constraints, etc.)
        subject_constraints = properties[property_id].get('subject_type_constraints', [])
        value_constraints = properties[property_id].get('value_type_constraints', [])

        if subject_constraints and subject_instance_type not in subject_constraints:
            return False, f"Subject '{subject}' type '{subject_instance_type}' does not satisfy the property '{property_label}' subject constraints: {subject_constraints}"

        if value_constraints and object_instance_type not in value_constraints:
            return False, f"Object '{obj}' type '{object_instance_type}' does not satisfy the property '{property_label}' value constraints: {value_constraints}"

        return True, "Correct"

# ---------- Metrics Calculation Module ----------
class MetricsCalculator:
    @staticmethod
    def calculate_overall_precision(responses, ground_truth, properties, property_constraints, subject_properties_dict):
        """Calculate precision, recall, and F1 score for responses against the ground truth."""
        total_precision = total_recall = correct_triples_cumulative = total_triples_cumulative = prompt_count = 0
        correction_prompts = []

        for _, row in responses.iterrows():
            subject = row['subject']
            response_triples = ast.literal_eval(row['triples'])
            subject_properties = subject_properties_dict.get(subject, set())

            errors, unused_declarations, correct_triples = [], [], 0

            for triple in response_triples:
                if triple[1] == 'instance of':
                    result, log = Validator.is_correct_instance_of(triple, properties, response_triples, property_constraints)
                    if result == "unused":
                        unused_declarations.append((triple, log))
                    elif not result:
                        errors.append((triple, log))
                    else:
                        correct_triples += 1
                else:
                    correct, log = Validator.is_correct_relation(triple, {t[0]: t[2] for t in response_triples if t[1] == 'instance of'}, 
                                                                 properties, property_constraints, subject_properties)
                    if not correct:
                        errors.append((triple, log))
                    else:
                        correct_triples += 1

            precision = 1 - (len(errors) / (len(response_triples) - len(unused_declarations)) 
                             if (len(response_triples) - len(unused_declarations)) > 0 else 1)

            ground_truth_row = ground_truth[ground_truth['subject'] == subject].iloc[0]
            gt_triples = TripleProcessor.parse_ground_truth_triples(ground_truth_row)
            results = MetricsCalculator.calculate_recall(response_triples, gt_triples, properties, property_constraints, subject_properties)

            recall = (results['gt_found'] + results['response_found']) / (results['gt_total'] + results['response_total'])

            correct_triples_cumulative += int(precision * len(response_triples))
            total_triples_cumulative += len(response_triples)

            total_precision += precision
            total_recall += recall
            prompt_count += 1

            if errors or unused_declarations:
                correction_prompt = PromptGenerator.generate_correction_prompt(subject, response_triples, errors, unused_declarations)
                correction_prompts.append((subject, correction_prompt))

        average_precision = total_precision / prompt_count if prompt_count > 0 else 0
        average_recall = total_recall / prompt_count if prompt_count > 0 else 0
        cumulative_precision = correct_triples_cumulative / total_triples_cumulative if total_triples_cumulative > 0 else 0

        f1_score = 2 * (average_precision * average_recall) / (average_precision + average_recall) if (average_precision + average_recall) > 0 else 0

        return {
            "average_precision": average_precision,
            "average_recall": average_recall,
            "cumulative_precision": cumulative_precision,
            "f1_score": f1_score,
            "correction_prompts": correction_prompts
        }

    @staticmethod
    def calculate_recall(response_triples, ground_truth_triples, properties, property_constraints, subject_properties):
        """Calculate recall and identify recall-related errors."""
        instances = {triple[0]: triple[2] for triple in response_triples if triple[1] == 'instance of'}
        
        correct_matches = total_gt_items = total_response_items = correct_response_items = 0
        recall_errors = []  # List to store recall-related errors

        for gt_triple in ground_truth_triples:
            for subject in gt_triple[0]:
                total_gt_items += 3
                found_match = False
                for response_triple in response_triples:
                    if response_triple[1] == 'instance of':
                        correct, log = Validator.is_correct_instance_of(response_triple, properties, response_triples, property_constraints)
                        if correct:
                            correct_matches += 3
                            found_match = True
                            break
                    else:
                        correct, log = Validator.is_correct_relation(response_triple, instances, properties, property_constraints, subject_properties)
                        if correct:
                            correct_matches += 3
                            found_match = True
                            break
                if not found_match:
                    recall_errors.append((gt_triple, f"Ground truth triple {gt_triple} not found in response"))

        for response_triple in response_triples:
            if response_triple[1] != 'instance of':
                total_response_items += 2
                subject_instance_type = instances.get(response_triple[0])
                object_instance_type = instances.get(response_triple[2])

                if subject_instance_type:
                    subject_valid, _ = Validator.is_valid_instance_type(response_triple[1], subject_instance_type, properties, property_constraints, True)
                    if subject_valid:
                        correct_response_items += 1
                    else:
                        recall_errors.append((response_triple, f"Invalid subject type for {response_triple[0]}"))
                if object_instance_type:
                    object_valid, _ = Validator.is_valid_instance_type(response_triple[1], object_instance_type, properties, property_constraints, False)
                    if object_valid:
                        correct_response_items += 1
                    else:
                        recall_errors.append((response_triple, f"Invalid object type for {response_triple[2]}"))

        return {
            "gt_found": correct_matches,
            "gt_total": total_gt_items,
            "response_found": correct_response_items,
            "response_total": total_response_items,
            "recall_errors": recall_errors  # Return the recall errors
        }

# ---------- Prompt Generation Module ----------
class PromptGenerator:
    @staticmethod
    def generate_correction_prompt(subject, response_triples, errors, unused_declarations):
        """Generate a correction prompt including errors and unused triples."""
        prompt = f"The following triples for the subject '{subject}' need attention:\n\n"

        if errors:
            prompt += "Errors:\n"
            for triple, error in errors:
                prompt += f"Triple: {triple}\nError: {error}\n\n"

        if unused_declarations:
            prompt += "Unused Declarations:\n"
            for triple, message in unused_declarations:
                prompt += f"Triple: {triple}\nNote: {message}\n\n"

        prompt += "Please provide a corrected set of triples for this subject, addressing the errors and unused declarations mentioned above."
        return prompt

# ---------- Main Function ----------
def main():
    """Main function to load data and calculate metrics."""
    # Load data
    property_constraints, properties, responses, ground_truth, subject_properties_dict = DataLoader.load_all_data()

    # Calculate precision
    precision_results = MetricsCalculator.calculate_overall_precision(
        responses, ground_truth, properties, property_constraints, subject_properties_dict
    )

    # Print results
    print(f"Average Precision: {precision_results['average_precision']:.2f}")
    print(f"Average Recall: {precision_results['average_recall']:.2f}")
    print(f"Cumulative Precision: {precision_results['cumulative_precision']:.2f}")
    print(f"F1 Score: {precision_results['f1_score']:.2f}")

    # Save correction prompts
    correction_df = pd.DataFrame(precision_results['correction_prompts'], columns=['subject', 'correction_prompt'])
    correction_df.to_csv('output/correction_prompts.csv', index=False)
    print("Correction prompts saved to 'correction_prompts.csv'")

def analyze_triples(response_file, ground_truth_file, properties_file, subjects_file):
    """Analyze triples by loading data from response, ground truth, and properties files."""
    # Load data
    property_constraints = DataLoader.load_json(properties_file)
    properties = DataLoader.load_properties(properties_file)
    responses = pd.read_csv(response_file)
    ground_truth = DataLoader.load_ground_truth(ground_truth_file)
    subjects_data = DataLoader.load_json(subjects_file)['data']
    subject_properties_dict = {subj: set(prop['label'] for prop in props) 
                               for subject_dict in subjects_data 
                               for subj, props in subject_dict.items()}

    # Calculate precision
    precision_results = MetricsCalculator.calculate_overall_precision(
        responses, ground_truth, properties, property_constraints, subject_properties_dict
    )

    return precision_results

# This allows the script to be imported without running the analysis
if __name__ == "__main__":
    print("This script is designed to be imported and used by other scripts.")
    print("To run the analysis, import this script and call the analyze_triples function.")

if __name__ == "__main__":
    main()
