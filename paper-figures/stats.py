import os
import pandas as pd

# Source directory where your data is stored
source_directory = "./collective"

# Results list to store calculated metrics for each configuration
results_list = []

# Iterate through configuration folders
for config_folder in os.listdir(source_directory):
    config_folder_path = os.path.join(source_directory, config_folder)

    if os.path.isdir(config_folder_path):
        # List all non-aggregated 'statistics.csv' files in the configuration folder
        csv_files = [f for f in os.listdir(config_folder_path) if f.endswith("_statistics.csv") and f != 'aggregated_statistics.csv']

        # Variables to track successful runs and total runs
        successful_runs = 0
        total_runs = len(csv_files)
        total_function_calls = 0

        # Iterate through non-aggregated 'statistics.csv' files
        for csv_file in csv_files:
            csv_file_path = os.path.join(config_folder_path, csv_file)
            df = pd.read_csv(csv_file_path, na_values='nan')

            # Check if the 'maximum fitness' column contains a 1 for successful run
            if 1 in df['maximum fitness'].values:
                successful_runs += 1

                # Find the earliest occurrence (in terms of generations) of a successful run
                earliest_successful_run = df.loc[df['maximum fitness'] == 1].iloc[0]

                # Get the function calls for the earliest successful run
                function_calls = earliest_successful_run['function calls']
                total_function_calls += function_calls

        # Calculate percentage of successful runs
        percentage_successful_runs = (successful_runs / total_runs) * 100 if total_runs > 0 else 0

        # Calculate mean function calls for successful runs
        mean_function_calls = total_function_calls / successful_runs if successful_runs > 0 else 0

        # Append results to the list
        results_list.append({
            'Configuration': config_folder,
            'Percentage_Successful_Runs': percentage_successful_runs,
            'Mean_Function_Calls': mean_function_calls
        })

# Create a DataFrame from the results list
results_df = pd.DataFrame(results_list)

# Save results to a new file
results_csv_path = "./results_summary.csv"
results_df.to_csv(results_csv_path, index=False)

print("Metrics calculation completed.")