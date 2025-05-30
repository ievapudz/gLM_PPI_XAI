#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex
import math
from scipy.ndimage import gaussian_filter1d

MATRIX_PATH = "./outputs/categorical_jacobians/glm2_cosine_post250521/"
MATRIX_SUFFIX = "fastCJ.npy"
PNG_PATH = "./outputs/visualisations/"
PNG_SUFFIX = "fastCJ_distribution.png"

parser = OptionParser()

parser.add_option("--input", "-i", dest="input",
    help="complex ID, whose categorical Jacobian matrix should be plotted.")

parser.add_option("--length-list", "-l", dest="length_list",
    help="list with lengths of the proteins.")

parser.add_option("--n-bins", "-n", dest="n_bins_urq",
    help="number of bins to use for URQ values", default=50)

parser.add_option("--text", "-t", dest="text",
    help="option to add more text in the plot.", action="store_true")

(options, args) = parser.parse_args()

def get_protein_length(length_list, complex_id):
    # Read the length list from the CSV file
    lengths = {}
    with open(length_list, 'r') as file:
        for i, line in enumerate(file):
            if(i):
                identifier, length = line.strip().split(',')
                identifier = identifier.replace('-', '_')
                lengths[identifier] = int(length)
   
    # Get the first protein id from the concatenated complex id
    first_protein_id = complex_id.split('_')[0].replace('-', '_')
    
    # Return the length of the first protein
    return lengths.get(first_protein_id, None)

def outlier_count(upper_right_quadrant, mode="mean_stddev", n=3, denominator=1e-8):
    if mode == "IQR":
        Q1 = np.percentile(upper_right_quadrant, 25)
        Q3 = np.percentile(upper_right_quadrant, 75)
        IQR = Q3-Q1
        threshold = Q3+1.5*IQR

    elif mode == "mean_stddev":
        m = np.mean(upper_right_quadrant)
        s = np.std(upper_right_quadrant)
        threshold = m+n*s

    elif mode == "ratio":
        threshold = 0.7
        upper_right_quadrant /= denominator

    count_above_threshold = np.sum(upper_right_quadrant > threshold)

    return count_above_threshold

def detect_ppi(array_2d, len1, padding=0.1):
    # Calculate the number of residues to ignore

    ignore_len1 = int(len1*padding)
    ignore_len2 = int((array_2d.shape[0]-len1)*padding)

    # Detecting the PPI signal in upper right quadrant of matrix
    upper_right_quadrant = array_2d[ignore_len1:len1-ignore_len1, len1+ignore_len2:-ignore_len2]
    quadrant_size = upper_right_quadrant.shape[0]*upper_right_quadrant.shape[1]

    # Detect outliers
    ppi = outlier_count(upper_right_quadrant, mode="mean_stddev", n=3)

    # Just a placeholder for the counting stage
    ppi_lab = 1 if(ppi) else 0
    
    # Detecting the PPI signal in upper right quadrant of matrix
    return ppi, ppi_lab

def create_figure(array_2d, output, len1):
    # array_2d - [NumPy array 2D] with logits

    padding = 0.1
    ignore_len1 = int(len1*padding)
    ignore_len2 = int((array_2d.shape[0]-len1)*padding)

    upper_right_quadrant = array_2d[ignore_len1:len1-ignore_len1, len1+ignore_len2:-ignore_len2]
    quadrant_flat = upper_right_quadrant.flatten()
    data_flat = array_2d.flatten()

    mean_val = np.mean(data_flat)
    std_val = np.std(data_flat)
    threshold_3sigma = mean_val + 3 * std_val
  
    font_size = 15 
    fig, ax = plt.subplots()

    plt.axvline(mean_val, color='#8676ab', linestyle='--', linewidth=2, label=f'μ: {mean_val:.3f}', zorder=0)
    plt.axvline(threshold_3sigma, color='#d85497', linestyle='--', linewidth=2,
               label=f'μ+nσ: {threshold_3sigma:.3f}', zorder=0)
    plt.text(mean_val+0.01, 100, "μ", verticalalignment='center', size=font_size)
    plt.text(threshold_3sigma+0.01, 100, "μ+nσ", verticalalignment='center', size=font_size)

    plt.hist(data_flat, bins=50, density=True, alpha=0.8, color='#8676ab', 
         edgecolor='black', linewidth=0.5, label=f'Full matrix')

    # Separate quadrant values into outliers and non-outliers
    quadrant_outliers = quadrant_flat[(quadrant_flat > threshold_3sigma)]
    quadrant_non_outliers = quadrant_flat[(quadrant_flat <= threshold_3sigma)]

    plt.hist(quadrant_flat, bins=int(options.n_bins_urq), density=True, alpha=1.0, color='#4b3381', 
         edgecolor='black', linewidth=0.5, 
         label=f'Upper-right quadrant')
    
    # Customize the plot
    plt.yscale('log')
    ax.tick_params(axis='both', which='major', labelsize=font_size)
    plt.xlabel('contact value', size=font_size)
    plt.ylabel('density', size=font_size)
    plt.title(f'Detection of outliers in upper-right quadrant', size=font_size)
    plt.legend(fontsize=font_size)

    # Add text box with statistics
    quadrant_outliers_pos = np.sum(quadrant_flat > threshold_3sigma)
    quadrant_outliers_total = quadrant_outliers_pos

    if(options.text):
        stats_text = (f'Full Array:\n'
                     f'  Size: {len(data_flat)}\n\n'
                     f'Upper-Right Quadrant:\n'
                     f'  Size: {len(quadrant_flat)}\n'
                     f'  Outliers beyond +nσ: {quadrant_outliers_pos} ({quadrant_outliers_pos/len(quadrant_flat)*100:.1f}%)\n')

        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                 fontsize=9)

    # Adjust layout to make space for the title
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to make space for the title

    # Save the figure with increased space for the title
    plt.savefig(output, bbox_inches='tight', pad_inches=0.5, dpi=300)

array_2d = np.load(f"{MATRIX_PATH}/{options.input}_{MATRIX_SUFFIX}")

length1 = get_protein_length(options.length_list, options.input)

ppi, ppi_lab = detect_ppi(array_2d, length1, padding=0.1)
print("PPI prediction: ", ppi, ppi_lab)

create_figure(array_2d, f"{PNG_PATH}/{options.input}_{PNG_SUFFIX}", length1)
