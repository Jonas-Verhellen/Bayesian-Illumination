import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the combined statistics from the CSV file
df = pd.read_csv("/home/jonasver/Documents/Code/Argenomic-GP/descriptor_rediscovery/combined_statistics.csv")

# Define finer intervals for function_calls for smoother interpolation
function_calls_intervals = np.arange(0, 1501, 50)

# Define a function to perform smoother interpolation within each run
def interpolate_within_run(run_group):
    # Ensure QD score at 0 function calls is set to 0 for each run
    run_group['quality_diversity'] = run_group['quality_diversity'].fillna(0)
    # Interpolate data within each run for the specified function call intervals
    interpolated_values = np.interp(function_calls_intervals, run_group['function_calls'], run_group['quality_diversity'])
    # Create DataFrame with interpolated values
    interpolated_df = pd.DataFrame({
        'Algorithm': run_group['Algorithm'].iloc[0],
        'Benchmark': run_group['Benchmark'].iloc[0],
        'function_calls': function_calls_intervals,
        'quality_diversity': interpolated_values
    })
    return interpolated_df

# Group by the "Run" column and interpolate QD for each run
interpolated_dfs = []
for _, run_group in df.groupby('Run'):
    interpolated_dfs.append(interpolate_within_run(run_group))

# Concatenate the interpolated DataFrames for all runs
interpolated_df = pd.concat(interpolated_dfs)

# Rename the benchmark names
benchmark_names = {0: 'USRCAT', 1: 'Zernike'}
interpolated_df['Benchmark'] = interpolated_df['Benchmark'].map(benchmark_names)

interpolated_df.to_csv("./interpolated.csv")
# Set seaborn theme and context
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=0.9)

# Create FacetGrid
g = sns.lineplot(x="function_calls", y="quality_diversity", hue="Algorithm", style="Benchmark", data=interpolated_df)
g.set(xlim=(0, 1501))
g.set(ylim=(0, 25))

# Set labels and title
plt.xlabel("Cumulative Function Calls")
plt.ylabel("Quality Diversity Score")
plt.title("Descriptor-based Redisocvery")

# Add legend with box
plt.legend(loc="upper left")

# Save the plot as a PDF with tight bounding box
fig = plt.gcf()
fig.set_size_inches(25, 10)
fig.savefig('qd_plot.pdf', bbox_inches='tight')

# Show the plot
plt.show()
