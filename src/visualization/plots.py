import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def generate_boxplot(df: pd.DataFrame):

    # Set custom color palette
    palette = sns.color_palette("husl", 3)

    # 1. Boxplot with Swarmplot
    fig = plt.figure(figsize=(10, 6))
    ax = sns.boxplot(data=df, palette=palette, width=0.5)
    plt.title(f'Error Metrics Distribution (n=288)', fontsize=14, pad=20)
    plt.ylabel('Error Value')
    plt.grid(axis='y', alpha=0.3)
    sns.despine()
    plt.tight_layout()

    #plt.savefig(PLOT_DIR+'/'+'boxplot.png')
    #plt.show()

    plt.close(fig)
    return fig

def generate_histogram(df: pd.DataFrame):

    # Set custom color palette
    palette = sns.color_palette("husl", 3)

    # 2. Histogram with KDE
    fig = plt.figure(figsize=(12, 6))
    for i, col in enumerate(df.columns):
        sns.histplot(df[col], kde=True, 
                    color=palette[i], 
                    label=f'{col} (μ={df[col].mean():.4f})',
                    bins=12, alpha=0.6)
        
    plt.title(f'Error Metrics Histogram (n=288)', fontsize=14, pad=20)
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(alpha=0.2)
    sns.despine()
    plt.tight_layout()

    #plt.savefig(PLOT_DIR+'/'+'histogram.png')
    #plt.show()

    plt.close(fig)
    return fig

def generate_density_plot(df: pd.DataFrame):

    # Set custom color palette
    palette = sns.color_palette("husl", 3)

    # 3. Combined Density Plot
    fig = plt.figure(figsize=(10, 5))
    for i, col in enumerate(df.columns):
        sns.kdeplot(df[col], color=palette[i], 
                    label=f'{col} (μ={df[col].mean():.4f})',
                    linewidth=2.5)
        
    plt.title(f'Error Metrics Density Comparison (n=288)', fontsize=14, pad=20)
    plt.xlabel('Error Value')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(alpha=0.2)
    sns.despine()
    plt.tight_layout()

    #plt.savefig(PLOT_DIR+'/'+'density_plot.png')
    #plt.show()

    plt.close(fig)
    return fig

def confusion_matrix_plot(y_true, y_pred):

    cm = confusion_matrix(y_true, y_pred)

    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True)

    plt.title(f'Comfusion Matrix', fontsize=14, pad=20)
    #plt.xlabel('Error Value')
    #plt.ylabel('Density')
    #plt.legend()
    #plt.grid(alpha=0.2)
    sns.despine()
    plt.tight_layout()

    plt.close(fig)
    return fig, cm
