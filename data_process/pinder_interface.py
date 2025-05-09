from pinder.core import get_pinder_location
from pinder.core.index.system import PinderSystem
from pinder.core.loader import structure
from biotite.structure import sasa, apply_residue_wise
import torch
from torch_geometric.nn import radius as georadius
import pandas as pd
import numpy as np
import os

PINDER_BASE_DIR="/scicore/home/schwede/durair0000/.local/share/"
os.environ["PINDER_BASE_DIR"] = PINDER_BASE_DIR
get_pinder_location()

def get_data(system_id, interface_threshold=8.0):
    system = PinderSystem(system_id)
    if system is None:
        return None
    seq_r, seq_l = system.native_R.sequence, system.native_L.sequence
    try:
        sasa_r = apply_residue_wise(
            system.native_R.atom_array, sasa(system.native_R.atom_array), np.nansum
        )
        sasa_l = apply_residue_wise(
            system.native_L.atom_array, sasa(system.native_L.atom_array), np.nansum
        )
    except Exception as e:
        print(e)
        return None
    r_coords = system.native_R.filter("atom_name", mask=["CA"]).coords
    l_coords = system.native_L.filter("atom_name", mask=["CA"]).coords
    if r_coords.shape[0] == len(seq_r) and l_coords.shape[0] == len(seq_l):
        pos_l, pos_r = georadius(
            torch.tensor(r_coords),
            torch.tensor(l_coords),
            r=interface_threshold,
            max_num_neighbors=10000,
        )
        return {
            "id": system_id,
            "pos_r": pos_r,
            "pos_l": pos_l,
            "seq_r": seq_r,
            "seq_l": seq_l,
            "sasa_r": sasa_r,
            "sasa_l": sasa_l,
        }
    return None

def get_atom_data(split, split_dir, pinder_base_dir=PINDER_BASE_DIR, interface_threshold=8.0):
    # Retrieving positions of the residues that are considered to be interacting with each other
    df = pd.read_csv(f"{split_dir}/{split}.txt", sep="\t")
    
    ids = {"R": [], "L": []}
    for i, j in df.iterrows():
        # Neglecting negatives
        if(df["label"][i] == 0): continue
        idR = df["protein1"][i]
        idL = df["protein2"][i]
        ids["R"].append(idR)
        ids["L"].append(idL)

    atom_data = []
    
    for i, id_ in enumerate(ids["R"]):
        try:
            struct_r = structure.Structure(
                f"{pinder_base_dir}/pinder/2024-02/pdbs/{id_}-R.pdb", 
                pinder_id=id_
            )
            struct_l = structure.Structure(
                f"{pinder_base_dir}/pinder/2024-02/pdbs/{ids['L'][i]}-L.pdb",
                pinder_id=ids['L'][i]
            )
            
            r_coords = struct_r.filter("atom_name", mask=["CA"]).coords
            l_coords = struct_l.filter("atom_name", mask=["CA"]).coords

            if(len(r_coords) != len(struct_r.sequence)):
                raise Exception(
                    f"Number of coordinates does not match the length of sequence {id_}-R"
                ) 
            if(len(l_coords) != len(struct_l.sequence)):
                raise Exception(
                    f"Number of coordinates does not match the length of sequence {ids['L'][i]}-L"
                ) 
        except Exception as e:
            print(e)
            continue

        sasa_r = apply_residue_wise(
            struct_r.atom_array, sasa(struct_r.atom_array), np.nansum
        )
        sasa_l = apply_residue_wise(
            struct_l.atom_array, sasa(struct_l.atom_array), np.nansum
        )
        
        dist_m = torch.cdist(torch.tensor(r_coords), torch.tensor(l_coords))

        pos_l, pos_r = georadius(
            torch.tensor(r_coords),
            torch.tensor(l_coords),
            r=interface_threshold,
            max_num_neighbors=10000,
        )

        pos_dists = []
        for j in range(len(pos_r)):
            pos_dists.append(dist_m[pos_r[j]][pos_l[j]].item())
        
        atom_data_el = {
            "id": f"{id_}-{ids['L'][i]}",
            "pos_r": pos_r,
            "pos_l": pos_l,
            "sasa_r": sasa_r,
            "sasa_l": sasa_l,
            "len_r": len(struct_r.sequence),
            "len_l": len(struct_l.sequence),
            "dist_matrix": dist_m,
            "pos_dists": pos_dists,
            "num_r_coords": len(r_coords),
            "num_l_coords": len(l_coords)
        }
        
        atom_data.append(atom_data_el)

    return atom_data

def get_contact_map(data_dict, padding=None):
    contact_map = np.zeros((data_dict["len_r"], data_dict["len_l"]))

    for i in range(len(data_dict["pos_r"])):
        contact_map[data_dict["pos_r"][i].item()][data_dict["pos_l"][i].item()] = 1

    if(padding):
        contact_map = contact_map[padding:-padding, padding:-padding]
    return contact_map

def get_cj_map(data_dict, cj_path, n, padding=None, biolm="gLM2"):
    id_for_file = data_dict["id"].translate(str.maketrans({'_': '-', '-': '_'}))
    cj = np.load(f"{cj_path}/{id_for_file}_fastCJ.npy")
    
    # Setting the threshold for the outlier count
    m = np.mean(cj)
    s = np.std(cj)
    threshold = m+n*s

    # Retrieving upper-right quadrant and skipping the tokens <+>
    if(biolm == "glm2"):
        upper_right_quadrant_cj = cj[1:data_dict["len_r"]+1, data_dict["len_r"]+2:]
        upper_right_quadrant_cm = np.where(upper_right_quadrant_cj > threshold, 1, 0)
    elif(biolm == "mint"):
        upper_right_quadrant_cj = cj[1:data_dict["len_r"]+1, data_dict["len_r"]+3:-1]
        upper_right_quadrant_cm = np.where(upper_right_quadrant_cj > threshold, 1, 0)
    
    if(padding):
        upper_right_quadrant_cj = upper_right_quadrant_cj[padding:-padding, padding:-padding]

    return upper_right_quadrant_cm, upper_right_quadrant_cj
