import pandas as pd
from src.triple_analysis import analyze_triples
from src.conversation import generate_triples
# from src.correction import apply_corrections
from src.utils import AzureOpenAIClient

# ---------- Main Script for Triple Analysis and Correction ----------
def main(cycles=1):
    """
    Main function to run the triple generation, analysis, and correction over a specified number of cycles.
    
    Args:
        cycles (int): Number of cycles to run the analysis. Defaults to 3.
    """
    # File paths
    response_file = 'data/anon/responses.csv'
    ground_truth_file = 'data/anon/ground_truth_anonymized.csv'
    properties_file = 'data/properties.json'
    subjects_file = 'data/subjects.json'

    # Initialize a list to store results for all cycles
    metrics_list = []

    # Loop through the specified number of cycles
    for cycle in range(cycles):
        print(f"Starting cycle {cycle + 1}")

        # Generate triples
        generate_triples(subjects_file, properties_file, output_file=response_file, ground_truth_file=ground_truth_file)

        # Analyze the generated triples
        results = analyze_triples(response_file, ground_truth_file, properties_file, subjects_file)

        # Collect metrics for the current cycle
        cycle_metrics = {
            'cycle': cycle + 1,
            'average_precision': results['average_precision'],
            'average_recall': results['average_recall'],
            'cumulative_precision': results['cumulative_precision'],
            'f1_score': results['f1_score']
        }

        # Append metrics for this cycle to the list
        metrics_list.append(cycle_metrics)

        # Print the metrics for the current cycle
        print(f"Cycle {cycle + 1} Results:")
        print(f"Average Precision: {results['average_precision']:.2f}")
        print(f"Average Recall: {results['average_recall']:.2f}")
        print(f"Cumulative Precision: {results['cumulative_precision']:.2f}")
        print(f"F1 Score: {results['f1_score']:.2f}")

        # Save correction prompts for the current cycle
        correction_df = pd.DataFrame(results['correction_prompts'], columns=['subject', 'correction_prompt'])
        correction_file = f'output/anon/correction_prompts_cycle_{cycle + 1}.csv'
        correction_df.to_csv(correction_file, index=False)
        print(f"Correction prompts saved to '{correction_file}'")

        # Apply corrections
        # apply_corrections(correction_file, response_file, properties_file, subjects_file)

    # Save the collected metrics for all cycles
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = 'output/anon/metrics_summary.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics for all cycles saved to '{metrics_file}'")

    print("All cycles completed.")

if __name__ == "__main__":
    main()
