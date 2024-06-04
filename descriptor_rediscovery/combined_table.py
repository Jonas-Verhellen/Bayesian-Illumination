import pandas as pd

# Load your CSV file
df = pd.read_csv("./final_statistics.csv")

# Group by Algorithm and Benchmark, and calculate mean and standard deviation
grouped_data = df.groupby(['Algorithm', 'Benchmark']).agg({'maximum_fitness': [('mean', 'mean'), ('std', 'std')],
                                                           'mean_fitness': [('mean', 'mean'), ('std', 'std')],
                                                           'quality_diversity': [('mean', 'mean'), ('std', 'std')]})

# Convert multi-index columns to single index
grouped_data.columns = [' '.join(col).strip() for col in grouped_data.columns.values]

# Extract mean and std values for each column
mean_values = grouped_data.filter(like='mean')
std_values = grouped_data.filter(like='std')

# Combine mean and std values with ± symbol
latex_table = mean_values.round(2).astype(str) + ' $\\pm$ ' + std_values.round(2).astype(str)

# Print or save LaTeX table
print(latex_table.to_latex(escape=False, column_format='lccccc'))
