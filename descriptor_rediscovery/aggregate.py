import os
import pandas as pd

# Directory containing algorithm folders
root_dir = './'
output_file = 'combined_statistics.csv'

# Initialize an empty list to store dataframes
dfs = []

# Traverse through each algorithm folder
for algorithm_folder in os.listdir(root_dir):
    algorithm_path = os.path.join(root_dir, algorithm_folder)
    
    # Skip non-directory entries
    if not os.path.isdir(algorithm_path):
        continue
    
    # Traverse through each timestamp folder
    for timestamp_folder in os.listdir(algorithm_path):
        timestamp_path = os.path.join(algorithm_path, timestamp_folder)
        
        # Skip non-directory entries
        if not os.path.isdir(timestamp_path):
            continue
        
        # Traverse through each benchmark folder
        for benchmark_folder in os.listdir(timestamp_path):
            benchmark_path = os.path.join(timestamp_path, benchmark_folder)
            
            # Skip non-directory entries
            if not os.path.isdir(benchmark_path):
                continue
            
            # Read statistics.csv file
            csv_file = os.path.join(benchmark_path, 'statistics.csv')
            if os.path.exists(csv_file):
                # Read CSV file and add algorithm, benchmark, and timestamp columns
                df = pd.read_csv(csv_file)
                df['Algorithm'] = algorithm_folder
                df['Benchmark'] = benchmark_folder
                
                # Append dataframe to the list
                dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Write combined dataframe to a new CSV file
combined_df.to_csv(output_file, index=False)

print(f"Combined statistics saved to {output_file}")