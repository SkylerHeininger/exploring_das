#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_daseg_results.py
Created 2/5/25 by DJ.
"""
# %%
# Import libraries
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

# Load results
test_group = 
control_group = 
test_group_dir = 
test_group_files = glob.glob(test_group_dir + '/*.csv')
control_group_dir = 
control_group_files = glob.glob(control_group_dir + '/*.csv')
print(f'Found {len(test_group_files)} files in {test_group_dir} and {len(control_group_files)} files in {control_group_dir}.')
# declare output folder
out_dir = 

# Create dataframe with ALL files that includes group
df_files = pd.DataFrame(columns=['file','group'])
df_files['file'] = test_group_files + control_group_files
df_files.loc[:len(test_group_files),'group'] = test_group
df_files.loc[len(test_group_files):,'group'] = 'control'

# Set up dataframes
df_word_label_counts = pd.DataFrame(columns=['participants','group','word_count'])
df_chunk_label_counts = pd.DataFrame(columns=['participants','group','chunk_count'])
# Loop through files
for file_index, in_file in enumerate(df_files['file']):
    # alert user every 20 files
    # if file_index<610:
    #     continue 
    if file_index%20==0:
        print(f'Processing file {file_index}/{len(df_files)}...')
    try:
        # Load results
        results = pd.read_csv(in_file)
        # Extract info
        participant = in_file.split('/')[-1].split('_wordlevel')[0]
        # Count
        word_count = len(results)
        chunk_count = results['Chunk'].max()
        # Count words per label
        word_label_counts = results['Proc DA Class'].value_counts()
        # Count chunks per label
        chunk_label_counts = results.groupby('Proc DA Class')['Chunk'].nunique()
        # Add to dataframe
        df_word_label_counts = pd.concat([df_word_label_counts, pd.DataFrame([word_label_counts])], ignore_index=True)
        df_chunk_label_counts = pd.concat([df_chunk_label_counts, pd.DataFrame([chunk_label_counts])], ignore_index=True)
        df_word_label_counts.loc[file_index,'participants'] = participant
        df_chunk_label_counts.loc[file_index,'participants'] = participant
        df_word_label_counts.loc[file_index,'group'] = df_files.loc[file_index,'group']
        df_chunk_label_counts.loc[file_index,'group'] = df_files.loc[file_index,'group']
        df_word_label_counts.loc[file_index,'word_count'] = word_count
        df_chunk_label_counts.loc[file_index,'chunk_count'] = chunk_count
    except Exception as e:
        print(e)
    
# Fill all missing values with 0
df_word_label_counts = df_word_label_counts.fillna(0)
df_chunk_label_counts = df_chunk_label_counts.fillna(0)
print('Done!')

# %%
# Save results
out_file = f'{out_dir}\daseg_word_label_counts.csv'
df_word_label_counts.to_csv(out_file)
print(f'Saved word-level results to {out_file}')
out_file = f'{out_dir}\daseg_chunk_label_counts.csv'
df_chunk_label_counts.to_csv(out_file)
print(f'Saved chunk-level results to {out_file}')

# %% Plot results
# Normalize by total number of words
df_word_label_counts_norm = df_word_label_counts.copy()
df_word_label_counts_norm.iloc[:,3:] = df_word_label_counts_norm.iloc[:,3:].divide(df_word_label_counts_norm['word_count'],axis=0)
df_chunk_label_counts_norm = df_chunk_label_counts.copy()
df_chunk_label_counts_norm.iloc[:,3:] = df_chunk_label_counts_norm.iloc[:,3:].divide(df_chunk_label_counts_norm['chunk_count'],axis=0)

# Sort columns by mean value in descending order
sorted_columns = df_word_label_counts_norm.iloc[:,3:].mean().sort_values(ascending=False).index
bar_width = 0.4  # Width of each bar
x = np.arange(len(sorted_columns))  # X positions for categories
groups = [test_group, control_group]
colors = ['blue', 'orange']


# Store p-values for each category
p_values_word = np.zeros(len(sorted_columns))
p_values_chunk = np.zeros(len(sorted_columns))

for column_index,column in enumerate(sorted_columns):
    # get word values
    group_a_word = df_word_label_counts_norm[df_word_label_counts_norm['group'] == groups[0]][column]
    group_b_word = df_word_label_counts_norm[df_word_label_counts_norm['group'] == groups[1]][column]    
    # Perform Mann-Whitney U test (non-parametric)
    _, p_val = mannwhitneyu(group_a_word, group_b_word, alternative='two-sided')
    p_values_word[column_index] = p_val
    # get chunk values
    # print(column, group_a_word, group_b_word, p_val)
    group_a_chunk = df_chunk_label_counts_norm[df_chunk_label_counts_norm['group'] == groups[0]][column]
    group_b_chunk = df_chunk_label_counts_norm[df_chunk_label_counts_norm['group'] == groups[1]][column]    
    # Perform Mann-Whitney U test (non-parametric)
    _, p_val = mannwhitneyu(group_a_chunk, group_b_chunk, alternative='two-sided')
    p_values_chunk[column_index] = p_val
    # print(column, group_a_chunk, group_b_chunk, p_val)

    
# Apply FDR correction
_, p_word_fdr, _, _ = multipletests(p_values_word, method='fdr_bh')
_, p_chunk_fdr, _, _ = multipletests(p_values_chunk, method='fdr_bh')

# Plot as bars
plt.figure(2,figsize=[13,10],clear=True)
plt.subplot(2,1,1)
# Plot word-level label distribution for each group separately
for group_index, (group, color) in enumerate(zip(groups, colors)):
    group_data = df_word_label_counts_norm[df_word_label_counts_norm['group'] == group]
    group_count = group_data.shape[0]
    print(group_data, group_data[sorted_columns].mean())
    plt.bar(x + (group_index - 0.5) * bar_width, group_data[sorted_columns].mean(), yerr=group_data[sorted_columns].std()/np.sqrt(group_count), width=bar_width, label=f'{group} (n={group_count})', color=color, alpha=0.7)
is_star_plotted = False
for i, p in enumerate(p_chunk_fdr):
    if p < 0.05:  # Mark significant comparisons
        # plt.text(x[i], max(group_data[sorted_columns].mean()) * 1.05, '*', ha='center', fontsize=12, color='red',label='p_{fdr}<0.05')
        if not is_star_plotted:
            plt.plot(x[i], max(group_data[sorted_columns].mean()) * 2, 'r*', label=r'$p_{fdr}<0.05$')
            is_star_plotted = True
        else:
            plt.plot(x[i], max(group_data[sorted_columns].mean()) * 2, 'r*')

plt.title('Word-level label distribution')
plt.legend()
plt.xlabel('Label')
plt.ylabel('Fraction of words (mean +/- ste)')
plt.grid(True)
plt.yscale('log')
# rotate x tick labels 45 degrees
plt.xticks(x,sorted_columns,rotation=90)#45,ha='right')
# plt.legend()
plt.subplot(2,1,2)
#sorted_columns_chunk = df_chunk_label_counts_norm.iloc[:,2:].mean().sort_values(ascending=False).index
# plt.bar(sorted_columns_chunk, df_chunk_label_counts_norm[sorted_columns_chunk].mean(), yerr=df_chunk_label_counts_norm[sorted_columns_chunk].std())
# plt.bar(sorted_columns, df_chunk_label_counts_norm[sorted_columns].mean(), yerr=df_chunk_label_counts_norm[sorted_columns].std())
# Plot chunk-level label distribution for each group separately
for group_index, (group, color) in enumerate(zip(groups, colors)):
    group_data = df_chunk_label_counts_norm[df_chunk_label_counts_norm['group'] == group]
    group_count = group_data.shape[0]
    print(group_data, group_data[sorted_columns].mean())
    plt.bar(x + (group_index - 0.5) * bar_width, group_data[sorted_columns].mean(), yerr=group_data[sorted_columns].std()/np.sqrt(group_count), width=bar_width, label=f'{group} (n={group_count})', color=color, alpha=0.7)
is_star_plotted = False
for i, p in enumerate(p_chunk_fdr):
    if p < 0.05:  # Mark significant comparisons
        # plt.text(x[i], max(group_data[sorted_columns].mean()) * 1.05, '*', ha='center', fontsize=12, color='red',label='p_{fdr}<0.05')
        if not is_star_plotted:
            plt.plot(x[i], max(group_data[sorted_columns].mean()) * 2, 'r*', label=r'$p_{fdr}<0.05$')
            is_star_plotted = True
        else:
            plt.plot(x[i], max(group_data[sorted_columns].mean()) * 2, 'r*')

plt.title('Chunk-level label distribution')
plt.legend()
plt.xlabel('Label')
plt.ylabel('Fraction of chunks (mean +/- ste)')
plt.grid(True)
plt.yscale('log')
# rotate x tick labels 45 degrees and align right
plt.xticks(x,sorted_columns,rotation=90)#45,ha='right')
# plt.legend()
plt.tight_layout()

# Save figure
out_file = f'{out_dir}\label_distributions.png'
plt.savefig(out_file)
print(f'Saved plot to {out_file}')
plt.show()
