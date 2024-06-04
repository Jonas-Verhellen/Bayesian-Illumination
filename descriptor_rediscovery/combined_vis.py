import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# Set seaborn theme and context
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=0.9)

expanded_df = pd.read_csv("./expanded.csv")
interpolated_df = pd.read_csv("./interpolated.csv")
interpolated_df["Descriptor"] = interpolated_df["Benchmark"] 
# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Set seaborn theme and context
sns.set_theme(style="whitegrid")
sns.set_context("talk", font_scale=0.9)

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# Plot 1: lmplot_figure with regression line
g = sns.scatterplot(x='fitness', y='descriptor_value', hue='Descriptor', data=expanded_df, alpha=0.5, ax=axes[0])
for _, sub_df in expanded_df.groupby('Descriptor'):
    sns.regplot(x='fitness', y='descriptor_value', data=sub_df, ax=axes[0], scatter=False)
g.set(xlim=(0, 0.8))
g.set(ylim=(0, 0.6))
axes[0].set_xlabel('Fingerprint Fitness')
axes[0].set_ylabel('Descriptor Fitness')
axes[0].set_title("Fingerprint - Descriptor Correlation")

# Plot 2: qd_plot
g = sns.lineplot(x="function_calls", y="quality_diversity", hue="Descriptor", style="Algorithm", data=interpolated_df, ax=axes[1], style_order=['GB-BI', 'GB-EPI'])
g.set(xlim=(0, 1501))
g.set(ylim=(0, 25))
axes[1].set_xlabel("Cumulative Function Calls")
axes[1].set_ylabel("QD Score")
axes[1].set_title("Descriptor-Based Rediscovery")
axes[1].legend(loc="upper left")

# Save the plot
fig.savefig('combined_plots.pdf', bbox_inches='tight')

# Show the plot
plt.show()
