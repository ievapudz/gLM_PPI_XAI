import argparse
import os
import torch
import hashlib
import time
from Encoding import Encoding

def main(args):

    devices = [int(s) for s in args.devices.split(",")]

    model_name_clean = args.model_name.split("/")[-1]

    encoding = Encoding(args.emb_type, 
        args.model_name, args.devices, 
        args.bs, args.max_seq_length, 
        args.pool_type
    )
    
    if(args.job_name):
        save_dir = f"{args.emb_dir}/{args.job_name}/{model_name_clean}_{encoding.get_type()}"
    else:
        hashlib.sha1().update(str(time.time()).encode("utf-8"))
        save_dir = f"{args.emb_dir}/{hashlib.sha1().hexdigest()[:10]}/{model_name_clean}"
        
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Running {model_name_clean} on {args.fasta}\n")

    embeddings = encoding.run(args.fasta, args.pair_list)

    # TODO: first check if there is already a file under the same name - do not overwrite if that is the case
    emb_file_name = f"{save_dir}/{os.path.splitext(os.path.basename(args.fasta))[0]}.pt"
    torch.save(embeddings, emb_file_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # General args
    parser.add_argument("--fasta", type=str, default=None)
    parser.add_argument("--pair_list", type=str, default=None)
    parser.add_argument("--emb_dir", type=str, default="./embeddings/")
    parser.add_argument("--job_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="tattabio/gLM2_650M")
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--devices", type=str, default="0")
    parser.add_argument("--pool_type", type=str, default="mean")
    parser.add_argument("--emb_type", type=str, default="single_seq") # can be: single_seq, joint_seq_joint_emb, joint_seq_sep_emb, sep_seq_sep_emb

    args = parser.parse_args()
    main(args)