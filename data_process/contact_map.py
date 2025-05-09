import pinder_interface as pi
import numpy as np
import os
import torch
import torch.nn.functional as F
from optparse import OptionParser

parser = OptionParser()

parser.add_option("--split", "-s", dest="split",
	default="test", help="split, for which to process the contact maps.")

parser.add_option("--biolm", "-b", dest="biolm",
    default="glm2", help="biological language model to use for categorical "+\
    "Jacobian. Choices: [glm2, esm2, mint].")

(options, args) = parser.parse_args()

def pad(matrix, max_L=512):
    M, N = matrix.shape
    pad_tot_M = max_L - M
    pad_left_M = pad_tot_M // 2
    pad_right_M = pad_tot_M - pad_left_M
    pad_tot_N = max_L - N
    pad_left_N = pad_tot_N // 2
    pad_right_N = pad_tot_N - pad_left_N

    return F.pad(matrix, (pad_left_N, pad_right_N, pad_left_M, pad_right_M), mode='constant', value=0.0)

SPLITS_DIR="/scicore/home/schwede/pudziu0000/projects/gLM/data/PINDER/eubacteria_5_1024_512_species_heterodimers/"
URQS_CM_PATH="/scicore/home/schwede/pudziu0000/projects/gLM/outputs/urqs/contact_maps/"
BIOLM="glm2"

CJ_PATH=f"/scicore/home/schwede/pudziu0000/projects/gLM/outputs/categorical_jacobians/{options.biolm}_cosine/"
URQS_CJ_PATH=f"/scicore/home/schwede/pudziu0000/projects/gLM/outputs/urqs/{options.biolm}_cosine/"

data = pi.get_atom_data(options.split, SPLITS_DIR)
print(f"Number of data points: {len(data)}")

os.makedirs(URQS_CJ_PATH, exist_ok=True)
os.makedirs(URQS_CM_PATH, exist_ok=True)

for i in range(len(data)):
    contact_map = pi.get_contact_map(data[i], padding=None)
    cj_contact_map, cj = pi.get_cj_map(data[i], CJ_PATH, n=3, padding=None, biolm=BIOLM)
    complex_id = data[i]['id'].translate(str.maketrans({'_': '-', '-': '_'}))
    contact_map = torch.from_numpy(contact_map).to(torch.float32)
    cj = torch.from_numpy(cj)
    contact_map = pad(contact_map)
    cj = pad(cj)
    torch.save(contact_map, f"{URQS_CM_PATH}/{complex_id}.pt")
    torch.save(cj, f"{URQS_CJ_PATH}/{complex_id}.pt")
    
