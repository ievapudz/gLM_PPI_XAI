#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex
import math
from scipy.ndimage import gaussian_filter1d

MATRIX_PATH = "./outputs/categorical_jacobians/"
MATRIX_SUFFIX = "fastCJ.npy"
PNG_PATH = "./outputs/visualisations/"
PNG_SUFFIX = "fastCJ.png"

parser = OptionParser()

parser.add_option("--input", "-i", dest="input",
    help="complex ID, whose categorical Jacobian matrix should be plotted.")

parser.add_option("--length-list", "-l", dest="length_list",
    help="list with lengths of the proteins.")

(options, args) = parser.parse_args()

def get_protein_length(length_list, complex_id):
    # Read the length list from the CSV file
    lengths = {}
    with open(length_list, 'r') as file:
        for line in file:
            identifier, length = line.strip().split(',')
            lengths[identifier] = int(length)
    
    # Get the first protein id from the concatenated complex id
    first_protein_id = complex_id.split('_')[0]
    
    # Return the length of the first protein
    return lengths.get(first_protein_id, None)

def apply_z_scores(array_2d, len1):
    quadrant = array_2d

    z_rows = (quadrant-quadrant.mean(axis=1, keepdims=True))/quadrant.std(axis=1, keepdims=True)
    z_cols = (quadrant-quadrant.mean(axis=0, keepdims=True))/quadrant.std(axis=0, keepdims=True)
    
    array_2d = (z_rows+z_cols)/2

    return array_2d

def apply_patching(array_2d, len1):
    import scipy.ndimage as ndimage

    quadrant = array_2d[:len1, len1:]
    quadrant = ndimage.gaussian_filter(quadrant, sigma=1)
    quadrant = ndimage.uniform_filter(quadrant, size=5)
    array_2d[:len1, len1:] = quadrant

    return array_2d

def outlier_count(upper_right_quadrant, mode="IQR", n=3, denominator=1e-8):
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

    # Normalize the data using the MinMax normalization and creating the colourmap

    cmap = plt.get_cmap("Blues")

    # Plot the heatmap
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    cax = ax.imshow(array_2d, cmap=cmap, aspect='auto', vmin=np.min(array_2d), 
        vmax=np.percentile(array_2d, 99), interpolation='none')

    # Add a red line at the last position of the first protein
    ax.axvline(x=len1-0.5, color='red', linestyle='--', linewidth=0.5)
    ax.axhline(y=len1-0.5, color='red', linestyle='--', linewidth=0.5)

    # Set x-axis title and ticks at the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Position i')
    ax.set_ylabel('Position j')
    ax.set_title(options.input)

    plt.colorbar(cax, ax=ax, label='Contact Value')

    # Adjust layout to make space for the title
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to make space for the title

    # Save the figure with increased space for the title
    plt.savefig(output, bbox_inches='tight', pad_inches=0.5, dpi=300)


array_2d = np.load(f"{MATRIX_PATH}/{options.input}_{MATRIX_SUFFIX}")

length1 = get_protein_length(options.length_list, options.input)

#corr_factors = np.load(f"outputs/entropy_factors/{options.input}_EntropyFactors.npy")
#array_2d = np.multiply(array_2d, corr_factors)
array_2d = apply_z_scores(array_2d, length1)
array_2d = apply_patching(array_2d, length1)

ppi, ppi_lab = detect_ppi(array_2d, length1, padding=0.1)
print("PPI prediction: ", ppi, ppi_lab)

create_figure(array_2d, f"{PNG_PATH}/{options.input}_{PNG_SUFFIX}", length1)
