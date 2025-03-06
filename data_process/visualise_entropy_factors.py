#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from optparse import OptionParser
import math

MATRIX_PATH = "./outputs/categorical_jacobians/"
MATRIX_SUFFIX = "fastCJ.npy"

FACTORS_PATH = "./outputs/entropy_factors/"
FACTORS_SUFFIX = "EntropyFactors.npy"

PNG_PATH = "./outputs/visualisations/"
PNG_SUFFIX = "CJEntropyFactorCorrection.png"

parser = OptionParser()

parser.add_option("--input", "-i", dest="input",
    help="complex ID, whose entropy matrix should be plotted.")

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
    return lengths[first_protein_id]
    
def create_figure(array_2d, output, len1):
    # array_2d - [NumPy array 2D] with deltas in entropies

    #cmap = plt.get_cmap("bwr")
    cmap = plt.get_cmap("Blues")

    # Plot the heatmap
    fig, ax = plt.subplots()
    #cax = ax.imshow(array_2d, cmap=cmap, aspect='auto', interpolation='none', vmin=-1, vmax=1)
    cax = ax.imshow(array_2d, cmap=cmap, aspect='auto', interpolation='none')

    # Add a red line at the last position of the first protein
    ax.axvline(x=len1-0.5, color='black', linestyle='--', linewidth=0.5)
    ax.axhline(y=len1-0.5, color='black', linestyle='--', linewidth=0.5)

    # Set x-axis title and ticks at the top
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    ax.set_xlabel('Position i in the sequence')
    ax.set_ylabel('Mutated position j')
    ax.set_title(options.input)

    plt.colorbar(cax, ax=ax, label='Corrected Contact Value')

    # Adjust layout to make space for the title
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust the rect parameter to make space for the title

    # Save the figure with increased space for the title
    plt.savefig(output, bbox_inches='tight', pad_inches=0.5, dpi=300)


# Loading the entropy matrix
m = np.load(f"{MATRIX_PATH}/{options.input}_{MATRIX_SUFFIX}")

# Loading the entropy factors
entropy_f = np.load(f"{FACTORS_PATH}/{options.input}_{FACTORS_SUFFIX}")

# Computing Hadamard product
array_2d = np.multiply(m, entropy_f)

length1 = get_protein_length(options.length_list, options.input)

create_figure(array_2d, f"{PNG_PATH}/{options.input}_{PNG_SUFFIX}", length1)
