import pandas as pd
import os
from pathlib import Path
from src.triple_analysis import analyze_triples
from src.conversation import generate_triples
from src.utils import AzureOpenAIClient

def process_chunk(chunk_folder, data_folder, output_folder, max_cycles=3):
    """
    Function to process a specific chunk for triple generation, analysis, and correction over multiple cycles.
    
    Args:
        chunk_folder (str): Path to the folder where the chunk is stored.
        data_folder (str): Path to the folder containing input data files.
        output_folder (str): Path to the folder for storing output files.
        max_cycles (int): Maximum number of cycles to run the analysis.
    """
    # Define file paths for the chunk
    subjects_file = Path(data_folder) / chunk_folder
    properties_file = os.path.join('data_anon', 'properties.json')
    ground_truth_file = os.path.join('data_anon', 'ground_truth.csv')

    chunk_output_dir = os.path.join(output_folder, chunk_folder)
    os.makedirs(chunk_output_dir, exist_ok=True)

    # Load or initialize final_responses.csv
    final_responses_file = os.path.join(chunk_output_dir, 'final_responses.csv')
    if not os.path.exists(final_responses_file):
        pd.DataFrame(columns=['subject', 'triples']).to_csv(final_responses_file, index=False)
    final_responses = pd.read_csv(final_responses_file)

    metrics_file = os.path.join(chunk_output_dir, 'metrics_summary.csv')
    if not os.path.exists(metrics_file):
        pd.DataFrame(columns=['cycle', 'average_precision', 'average_recall', 'cumulative_precision', 'f1_score']).to_csv(metrics_file, index=False)

    # Initialize a list to store metrics for all cycles
    metrics_list = []

    # Determine the starting cycle by checking for correction prompts
    existing_correction_files = [f for f in os.listdir(chunk_output_dir) if f.startswith('correction_prompts_cycle_')]
    start_cycle = len(existing_correction_files) + 1

    # Loop through the cycles
    for cycle in range(start_cycle, max_cycles + 1):
        print(f"Starting cycle {cycle} for chunk {chunk_folder}")

        # Define response file for this cycle
        response_file = os.path.join(chunk_output_dir, f'responses_cycle_{cycle}.csv')

        # Load correction prompts for the current cycle, if they exist
        correction_file = os.path.join(chunk_output_dir, f'correction_prompts_cycle_{cycle-1}.csv')
        correction_prompts = None
        if os.path.exists(correction_file):
            correction_prompts = pd.read_csv(correction_file)
            print(f"Loaded correction prompts from {correction_file}")
        else:
            print(f"No correction prompts found for cycle {cycle}")

        # Generate triples for the current cycle
        results = generate_triples(subjects_file, properties_file, response_file,
                                   ground_truth_file, chunk_output_dir, correction_file if correction_prompts is not None else None, cycle)

        # Check if the response file was created
        if not os.path.exists(response_file):
            print(f"No responses generated for cycle {cycle}. Ending processing.")
            break

        # Load and update final responses
        current_responses = pd.read_csv(response_file)
        final_responses = final_responses[~final_responses['subject'].isin(current_responses['subject'])]
        final_responses = pd.concat([final_responses, current_responses], ignore_index=True)
        final_responses.to_csv(final_responses_file, index=False)
        print(f"Updated final responses saved to '{final_responses_file}'")

        # Analyze the generated triples using final_responses.csv
        results = analyze_triples(final_responses_file, ground_truth_file, properties_file, subjects_file)

        # Save and print metrics for the current cycle
        cycle_metrics = {'cycle': cycle, 'average_precision': results['average_precision'],
                         'average_recall': results['average_recall'], 'cumulative_precision': results['cumulative_precision'], 'f1_score': results['f1_score']}
        print(f"Cycle {cycle} Results: Average Precision: {results['average_precision']:.2f}, "
              f"Average Recall: {results['average_recall']:.2f}, 'cumulative_precision': {results['cumulative_precision']:.2f}, F1 Score: {results['f1_score']:.2f}")

        # Append metrics to metrics_summary.csv
        metrics_df = pd.DataFrame([cycle_metrics])
        metrics_df.to_csv(metrics_file, mode='a', header=False, index=False)
        print(f"Metrics for cycle {cycle} appended to '{metrics_file}'")

        # Save correction prompts for the next cycle
        if 'correction_prompts' in results:
            correction_df = pd.DataFrame(results['correction_prompts'], columns=['subject', 'correction_prompt'])
            next_correction_file = os.path.join(chunk_output_dir, f'correction_prompts_cycle_{cycle}.csv')
            correction_df.to_csv(next_correction_file, index=False)
            print(f"Correction prompts saved to '{next_correction_file}'")

    print(f"Metrics for all cycles saved to '{metrics_file}'")
    print(f"All cycles for chunk {chunk_folder} completed.")

def main(max_cycles=3, data_folder='data_anon/chunks', chunk_folder_prefix='chunk_', output_folder='output_anon', specific_chunk='chunk_79.json'):
    """
    Main script for processing chunks of data.

    Args:
        max_cycles (int): Maximum number of cycles for processing.
        data_folder (str): Base folder containing the data chunks.
        chunk_folder_prefix (str): Prefix to identify chunks.
        output_folder (str): Folder to store output results.
        specific_chunk (str): Specific chunk to process, if provided.
    """
    if specific_chunk:
        chunk_path = os.path.join(data_folder, specific_chunk)
        if os.path.exists(chunk_path):
            print(f"Processing specific chunk: {specific_chunk}")
            process_chunk(specific_chunk, data_folder=data_folder, output_folder=output_folder, max_cycles=max_cycles)
            print(f"Completed processing for {specific_chunk}")
        else:
            print(f"Error: Chunk {specific_chunk} does not exist in {data_folder}")
        return

    # Uncomment the following lines to process all chunks
    # chunk_folders = [f for f in os.listdir(data_folder) if f.startswith(chunk_folder_prefix) and os.path.isdir(os.path.join(data_folder, f))]
    # for chunk_folder in chunk_folders:
    #     process_chunk(chunk_folder, data_folder=data_folder, output_folder=output_folder, max_cycles=max_cycles)
    #     print(f"Completed processing for {chunk_folder}")

if __name__ == "__main__":
    main()