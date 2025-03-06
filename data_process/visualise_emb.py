#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
from matplotlib.colors import Normalize, LinearSegmentedColormap, to_hex
import math
from scipy.ndimage import gaussian_filter1d

MATRIX_PATH = "./outputs/embeddings_matrices/"
MATRIX_SUFFIX = "fastEmb.npy"
PNG_PATH = "./outputs/visualisations/"
PNG_SUFFIX = "fastEmb.png"

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
    
def create_figure(array_2d, output, len1):
    # array_2d - [NumPy array 2D] with logits

    # Normalize the data using the MinMax normalization and creating the colourmap
    vmin = array_2d.min()
    vmax = np.percentile(array_2d, 99)
    norm = Normalize(vmin=vmin, vmax=vmax)

    cmap = plt.get_cmap("Blues")
    blues = [cmap(i) for i in np.linspace(0, 1, 256)]
    custom_cmap = LinearSegmentedColormap.from_list("custom_blues", blues, N=256)

    # Plot the heatmap
    fig, ax = plt.subplots()
    ax.invert_yaxis()
    cax = ax.imshow(array_2d, cmap=custom_cmap, norm=norm, aspect='auto', interpolation='none')

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

# Example usage
length1 = get_protein_length(options.length_list, options.input)

create_figure(array_2d, f"{PNG_PATH}/{options.input}_{PNG_SUFFIX}", length1)
