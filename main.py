import pandas as pd
from src.triple_analysis import analyze_triples
from src.conversation import generate_triples
# from src.correction import apply_corrections
from src.utils import load_api_keys, set_openai_api_configurations

def main(cycles=3):
    # File paths
    response_file = 'data/responses.csv'
    ground_truth_file = 'data/ground_truth.csv'
    properties_file = 'data/properties.json'
    subjects_file = 'data/subjects.json'

    # Load API keys and set configurations
    load_api_keys("config/api_keys.json")
    set_openai_api_configurations()

    # Initialize a list to store results for all cycles
    metrics_list = []

    for cycle in range(cycles):
        print(f"Starting cycle {cycle + 1}")

        # Generate triples
        # generate_triples(subjects_file, properties_file, output_file=response_file)

        # Analyze triples
        results = analyze_triples(response_file, ground_truth_file, properties_file, subjects_file)

        # Collect metrics
        cycle_metrics = {
            'cycle': cycle + 1,
            'average_precision': results['average_precision'],
            'average_recall': results['average_recall'],
            'cumulative_precision': results['cumulative_precision'],
            'f1_score': results['f1_score']
        }
        
        # Append metrics for this cycle to the list
        metrics_list.append(cycle_metrics)

        # Print metrics
        print(f"Cycle {cycle + 1} Results:")
        print(f"Average Precision: {results['average_precision']:.2f}")
        print(f"Average Recall: {results['average_recall']:.2f}")
        print(f"Cumulative Precision: {results['cumulative_precision']:.2f}")
        print(f"F1 Score: {results['f1_score']:.2f}")

        # Save correction prompts
        correction_df = pd.DataFrame(results['correction_prompts'], columns=['subject', 'correction_prompt'])
        correction_file = f'output/correction_prompts_cycle_{cycle + 1}.csv'
        correction_df.to_csv(correction_file, index=False)
        print(f"Correction prompts saved to '{correction_file}'")

        # Apply corrections
        # apply_corrections(correction_file, response_file, properties_file, subjects_file)

    # Save the collected metrics for all cycles
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = 'output/metrics_summary.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"Metrics for all cycles saved to '{metrics_file}'")

    print("All cycles completed.")

if __name__ == "__main__":
    main()
