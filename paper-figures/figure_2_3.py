import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec



# Source directory where your reformatted data is stored
source_directory = "./collective"

# Get list of configuration folders and sort them by number
config_folders = sorted([folder for folder in os.listdir(source_directory) if os.path.isdir(os.path.join(source_directory, folder))], key=lambda x: int(x))

# Define the mapping of configuration names to numbers
config_number_to_name = {
    0: 'ECFP4 - Mean',
    1: 'ECFP4 - UCB',
    2: 'ECFP4 - EI',
    3: 'ECFP4 - logEI',
    4: 'ECFP6 - Mean',
    5: 'ECFP6 - UCB',
    6: 'ECFP6 - EI',
    7: 'ECFP6 - logEI',
    8: 'FCFP4 - Mean',
    9: 'FCFP4 - UCB',
    10: 'FCFP4 - EI',
    11: 'FCFP4 - logEI',
    12: 'FCFP6 - Mean',
    13: 'FCFP6 - UCB',
    14: 'FCFP6 - EI',
    15: 'FCFP6 - logEI',
    16: 'RDFP - Mean',
    17: 'RDFP - UCB',
    18: 'RDFP - EI',
    19: 'RDFP - logEI',
    20: 'APFP - Mean',
    21: 'APFP - UCB',
    22: 'APFP - EI',
    23: 'APFP - logEI',
    24: 'TTFP - Mean',
    25: 'TTFP - UCB',
    26: 'TTFP - EI',
    27: 'TTFP - logEI',
    28: 'SMI - Mean',
    29: 'SMI - UCB',
    30: 'SMI - EI',
    31: 'SMI - logEI',
    32: 'SLF - Mean',
    33: 'SLF - UCB',
    34: 'SLF - EI',
    35: 'SLF - logEI',

}

custom_order = [
    0, 4, 8, 12, 16, 20, 24, 28, 32,  # Mean configurations
    1, 5, 9, 13, 17, 21, 25, 29, 33, # UCB configurations
    2, 6, 10, 14, 18, 22, 26, 30, 34, # EI configurations
    3, 7, 11, 15, 19, 23, 27, 31, 35 # logEI configurations
]

# Create a custom mapping for the left-side plot
custom_mapping = {i: config_number_to_name[i] for i in custom_order}



sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=0.9)

# Iterate through each variable
variables = ['maximum fitness', 'mean fitness', 'quality diversity score', 'coverage', 'max_err', 'mse', 'mae']
titles = ['Maximum Fitness', 'Mean Fitness', 'QD Score', 'Coverage', 'Max Error', 'MSE', 'MEA']

variable_titles_mapping = {
    "('maximum fitness', 'mean')": 'Maximum Fitness',
    "('mean fitness', 'mean')": 'Mean Fitness',
    "('quality diversity score', 'mean')": 'QD Score',
    "('coverage', 'mean')": 'Coverage',
    "('max_err', 'mean')": 'Max Error',
    "('mse', 'mean')": 'MSE',
    "('mae', 'mean')": 'MEA',
}

data = pd.DataFrame()

# # Iterate through each configuration folder
# for config_folder in config_folders:
#     config_number = int(config_folder)
#     config_name = config_number_to_name[config_number]

#     # Load the data for the current configuration
#     config_data = pd.read_csv(os.path.join(source_directory, config_folder, 'aggregated_statistics.csv')).tail(1)

#     # Add configuration information to the data
#     config_data['config'] = config_name

#     # Append the data to the main DataFrame
#     data = data.append(config_data, ignore_index=True)

for config_number in custom_order:
    config_name = custom_mapping[config_number]
    config_folder = str(config_number)
    config_data = pd.read_csv(os.path.join(source_directory, config_folder, 'aggregated_statistics.csv')).tail(1)
    config_data['config'] = config_name
    data = data.append(config_data, ignore_index=True)

def rename_columns(df, column_mapping):
    return df.rename(columns=column_mapping).copy()

newdata = rename_columns(data, variable_titles_mapping)
# Make the PairGrid
g = sns.PairGrid(newdata,
                 x_vars=[title for title in titles],
                 y_vars=["config"],
                 height=10, aspect=.25)


# Draw a dot plot using the stripplot function
g.map(sns.stripplot, size=8, orient="h", jitter=False, palette="flare_r", linewidth=2, edgecolor="w", alpha=1)
#g.map(sns.pointplot, markers='o', linestyles='', dodge=True, palette="flare_r", scale=0.6)

# Use semantically meaningful titles for the columns
import numpy as np
for ax, title, variable in zip(g.axes.flat, titles, variables):
    # Set a different title for each axes
    # Make the grid horizontal instead of vertical
    ax.xaxis.grid(False)
    ax.yaxis.grid(True)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)

    if variable == 'maximum fitness':
        ax.set(xlim=(0.57, 1.03))
        ax.set_xticks([0.6, 1])
    elif variable == 'mean fitness':
        ax.set(xlim=(0.39, 0.51))
        ax.set_xticks([0.4, 0.5])
    elif variable == 'quality diversity score':
        ax.set(xlim=(29, 52))
        ax.set_xticks([30, 50])
    elif variable == 'coverage':
        ax.set(xlim=(48, 82))
        ax.set_xticks([50, 80])
    elif variable == 'max_err':
        ax.set(xlim=(-0.05, 0.55))
        ax.set_xticks([0, 0.5])
    elif variable == 'mse':
        ax.set(xlim=(-0.005, 0.055))
        ax.set_xticks([0, 0.05])
    elif variable == 'mae':
        ax.set(xlim=(-0.015, 0.215))
        ax.set_xticks([0, 0.2])

    # Make x-axis ticks more readable
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=-90, ha='left')  # Adjust rotation angle and horizontal alignment

    # Optionally, adjust other tick parameters for better readability
    # ax.tick_params(axis='x', labelsize=11, width=1, length=5)

    std_devs = data[f"('{variable}', 'std')"]
    mean_values = data[f"('{variable}', 'mean')"]
    if variable == 'maximum fitness':
        xerr_upper = np.minimum(std_devs, np.max(mean_values) - mean_values)
    else:
        xerr_upper = std_devs
    xerr_lower =  np.minimum(std_devs,  mean_values)
    ax.errorbar(x=data[f"('{variable}', 'mean')"], y=np.arange(len(data)), xerr=(xerr_lower, xerr_upper), fmt='none', ecolor='steelblue', alpha=0.9, capsize=4)
    #ax.xaxis.set_major_locator(MaxNLocator(nbins=4))

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

sns.despine(left=True, bottom=False)
plt.savefig('intervals.pdf', bbox_inches='tight')

# Calculate correlation matrix
correlation_matrix = data.corr()

correlation_matrix.to_csv('correlation_matrix.csv')

# Extract acquisition function information from configuration names
fingerprints  = [name.split(' - ')[0] for name in config_number_to_name.values()]
acquisition_functions = [name.split(' - ')[1] for name in config_number_to_name.values()]

# Initialize an empty DataFrame to store final values
variables = [f"('{variable}', 'mean')" for variable in ['maximum fitness', 'mean fitness', 'quality diversity score', 'coverage', 'function calls', 'max_err', 'mse', 'mae']]
final_values_df = pd.DataFrame(index=config_folders, columns=variables)
# Iterate through each configuration folder
for config_folder in config_folders:
    config_folder_path = os.path.join(source_directory, config_folder)

    # Read reformatted statistics CSV for the configuration
    aggregated_csv_path = os.path.join(config_folder_path, 'aggregated_statistics.csv')
    df = pd.read_csv(aggregated_csv_path)

    # Get the final values for each variable
    final_values = df.iloc[-1][variables]

    # Add values to the final_values_df
    final_values_df.loc[config_folder] = final_values
    final_values_df["fingerprints"] = fingerprints
    final_values_df["acquisition_functions"] = acquisition_functions


for variable in variables:
    final_values_df[variable] = final_values_df[variable].astype(float)

# Create a dictionary to store DataFrames for each variable
variable_dataframes = {}
reshaped_df = final_values_df.pivot(index='acquisition_functions', columns='fingerprints')

# Iterate through each variable and create a DataFrame
for variable in variables:
    variable_dataframes[variable] = reshaped_df[variable]

# Create the right side of the figure with heatmaps
fig = plt.figure(figsize=(25,10))  # Adjust the size as needed

# Create a gridspec to control the layout
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.1)


additional_data_path = '/home/jonasver/Documents/Code/Argenomic-GP/fingerprint_tr_run/results_summary.csv'  # Replace with the actual path
additional_data_df = pd.read_csv(additional_data_path)
additional_data_df['config_name'] = additional_data_df['Configuration'].map(config_number_to_name)
additional_data_df[['fingerprints', 'acquisition_functions']] = additional_data_df['config_name'].str.split(' - ', expand=True)
additional_data_df['Mean_Function_Calls'] = additional_data_df['Mean_Function_Calls'].replace(0, 5000)

# Heatmap 1
ax_heatmap1 = plt.subplot(gs[0])
df = variable_dataframes[variables[0]]
sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, vmin=df.values.min(), vmax=df.values.max(), center=df.values.mean(), cmap=sns.color_palette("light:b", as_cmap=True), cbar=True)
ax_heatmap1.set_xlabel('')
ax_heatmap1.set_ylabel('')
ax_heatmap1.set_title('Maximum Fitness (Mean Values)')

# Heatmap 2
ax_heatmap1 = plt.subplot(gs[2])
df = variable_dataframes[variables[5]]
sns.heatmap(df, annot=True, fmt=".2f", linewidths=.5, vmin=df.values.min(), vmax=df.values.max(), center=df.values.mean(), cmap=sns.color_palette("light:b_r", as_cmap=True), cbar=True)
ax_heatmap1.set_xlabel('')
ax_heatmap1.set_ylabel('')
ax_heatmap1.set_title('Maximum Error (Mean Values)')

# Heatmap 3 (Percentage_Successful_Runs)
ax_heatmap2 = plt.subplot(gs[1])
df_percentage_successful_runs = additional_data_df.pivot(index='acquisition_functions', columns='fingerprints', values='Percentage_Successful_Runs')
sns.heatmap(df_percentage_successful_runs, annot=True, fmt=".2f", linewidths=.5, vmin=df_percentage_successful_runs.values.min(),
            vmax=df_percentage_successful_runs.values.max(), center=df_percentage_successful_runs.values.mean(),
            cmap=sns.color_palette("light:b", as_cmap=True), cbar=True)
ax_heatmap2.set_xlabel('')
ax_heatmap2.set_ylabel('')
ax_heatmap2.set_title('Rediscovery Rate (Percentage)')

# Heatmap 4 (Mean_Function_Calls)
ax_heatmap3 = plt.subplot(gs[3])
df_mean_function_calls = additional_data_df.pivot(index='acquisition_functions', columns='fingerprints', values='Mean_Function_Calls')
sns.heatmap(df_mean_function_calls, annot=True, fmt=".0f", linewidths=.5, vmin=df_mean_function_calls.values.min(),
            vmax=df_mean_function_calls.values.max(), center=df_mean_function_calls.values.mean(),
            cmap=sns.color_palette("light:b_r", as_cmap=True), cbar=True)
ax_heatmap3.set_xlabel('')
ax_heatmap3.set_ylabel('')
ax_heatmap3.set_title('Fitness Calls (Mean Values)')
fig.savefig('stacked_heatplots.pdf', bbox_inches='tight')


#plt.show()
