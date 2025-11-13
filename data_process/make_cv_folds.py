import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
import argparse

def make_cv_folds(train_file, val_file, k=5, output_dir="cv_folds"):
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_df = pd.read_csv(train_file, sep="\t")
    val_df = pd.read_csv(val_file, sep="\t")

    # Stratified split of training and validation into k parts
    skf_train = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    skf_val = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)

    train_divisions = []
    val_divisions = []

    for _, idx in skf_train.split(train_df, train_df["label"]):
        train_divisions.append(train_df.iloc[idx])

    for _, idx in skf_val.split(val_df, val_df["label"]):
        val_divisions.append(val_df.iloc[idx])

    # Create k folds
    for i in range(k):
        # For simplicity, use 4 training divisions and 1 validation division per fold
        train_parts = [train_divisions[j] for j in range(k) if j != i]
        train_fold = pd.concat(train_parts, axis=0).reset_index(drop=True)
        val_fold = val_divisions[i].reset_index(drop=True)

        # Save to txt
        train_path = os.path.join(output_dir, f"{i}/train.txt")
        val_path = os.path.join(output_dir, f"{i}/validate.txt")
        
        if not os.path.exists(f"{output_dir}/{i}/"):
            os.makedirs(f"{output_dir}/{i}/")

        train_fold.to_csv(train_path, sep="\t", index=False)
        val_fold.to_csv(val_path, sep="\t", index=False)

        print(f"✅ Created fold {i}: {len(train_fold)} train, {len(val_fold)} val samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate cross-validation folds from training and validation splits.")
    parser.add_argument("--train", required=True, help="Path to training txt file.")
    parser.add_argument("--val", required=True, help="Path to validation txt file.")
    parser.add_argument("--k", type=int, default=5, help="Number of folds (default: 5).")
    parser.add_argument("--out", default="cv_folds", help="Output directory for CV folds.")
    args = parser.parse_args()

    make_cv_folds(args.train, args.val, args.k, args.out)

