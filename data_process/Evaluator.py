import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import matplotlib as mpl

class Evaluator:
    def __init__(self, base_dir, representations, biolms, dev_stage, num_folds, 
                 out_dir, hyperparam, hyperparam_value):
        self.base_dir = base_dir
        self.representations = representations
        self.biolms = biolms
        self.dev_stage = dev_stage
        self.num_folds = num_folds
        # TODO: set variable self.folds based on dev_stage and num_folds
        if(dev_stage == "CV"):
            self.folds = range(self.num_folds)
        elif(dev_stage == "train_validate"):
            self.folds = [dev_stage]
        self.hyperparam = hyperparam
        self.hyperparam_value = hyperparam_value
        if(hyperparam and hyperparam_value):
            self.out_dir = f"{out_dir}/{hyperparam}_{hyperparam_value}/{dev_stage}/"
        else:
            self.out_dir = f"{out_dir}/{dev_stage}/"
    
    def initialise_df(self):
        representation_col = []
        biolm_col = []
        fold_col = []
        for representation in self.representations:
            for biolm in self.biolms:
                if(representation == "joint_input_separate_pooling" and biolm == "MINT"): continue
                biolm_col += [biolm]*self.num_folds
                representation_col += [representation]*self.num_folds
                for n in range(self.num_folds):
                    fold_col.append(n)
    
        df = pd.DataFrame({
            "representation": representation_col,
            "biolm": biolm_col,
            "fold": fold_col
        })
        return df

    # Helper functions to compute metrics from the confusion matrix
    def get_accuracy(self, row):
        tp = row["tp"].values[0]
        tn = row["tn"].values[0]
        fp = row["fp"].values[0]
        fn = row["fn"].values[0]
        denom = tp + tn + fp + fn
        metric_value = (tp + tn) / denom if denom > 0 else 0.0
        return metric_value
    
    def get_f1(self, row):
        tp = row["tp"].values[0]
        tn = row["tn"].values[0]
        fp = row["fp"].values[0]
        fn = row["fn"].values[0]
        denom = (2*tp + fp + fn)
        metric_value = 2*tp / denom if denom > 0 else 0.0
        return metric_value
    
    def get_metric_of_interest_values(self, metric_of_interest):
        experiment_results = {}
        experiment_epochs = {}
    
        metrics_arr = []
        best_epochs_arr = []
        for representation in self.representations:
            for biolm in self.biolms:
                fold_metrics = []
                for fold_num in self.folds:
                    # Define paths
                    fold_path = f"{self.base_dir}/{representation}/{biolm}/{fold_num}"
                    ckpt_path = glob(f"{fold_path}/checkpoints/model-epoch=*.ckpt")
            
                    if not ckpt_path:
                        print(f"No checkpoint found for {representation}/{biolm}, fold {fold_num}")
                        continue
                    
                    # Extract best epoch
                    match = re.search(r"model-epoch=(\d+)\.ckpt", os.path.basename(ckpt_path[0]))
                    if not match:
                        print(f"Could not extract epoch from checkpoint for {representation}/{biolm}, fold {fold_num}")
                        continue
                    best_epoch = int(match.group(1))
                    best_epochs_arr.append(best_epoch)
            
                    # Read metrics
                    metrics_file = os.path.join(fold_path, "metrics.csv")
                    if not os.path.isfile(metrics_file):
                        print(f"No metrics.csv found for exp {representation}/{biolm}, fold {fold_num}")
                        continue
                    df = pd.read_csv(metrics_file)
            
                    # Drop the first duplicate line of epoch 0
                    df = df[~((df.epoch == 0) & (df.index == df[df.epoch == 0].index.min()))]
            
                    # Get metric at best epoch
                    row = df[df.epoch == best_epoch]
                    if row.empty:
                        print(f"No data for best epoch {best_epoch} in {representation}/{biolm}, fold {fold_num}")
                        continue
        
                    if metric_of_interest != "accuracy" and metric_of_interest != "f1":
                        metric_value = row[metric_of_interest].values[0]
                    else:
                        required_cols = {"tn", "fp", "fn", "tp"}
                        if not required_cols.issubset(row.columns):
                            print(f"Missing columns for {metric_of_interest} in {representation}/{biolm}, fold {fold_num}")
                            continue
                        if metric_of_interest == "accuracy": metric_value = self.get_accuracy(row)
                        if metric_of_interest == "f1": metric_value = self.get_f1(row)
                    fold_metrics.append(metric_value)
                    metrics_arr.append(metric_value)
            
                if fold_metrics:
                    experiment_results[f"{representation} {biolm}"] = fold_metrics
                    
        return (experiment_results, experiment_epochs, metrics_arr, best_epochs_arr)
    
    def get_performance_stats(self, results):
        means = []
        stderrs = []
        labels = []
        
        for model_type, metrics in results.items():
            mean_val = sum(metrics) / len(metrics)
            stderr_val = pd.Series(metrics).std()/len(metrics)**(1/2)
            means.append(mean_val)
            stderrs.append(stderr_val)
            labels.append(f"{model_type}")
    
        return means, stderrs, labels
    
    def plot_stats(self, means, stderrs, labels, metric):
        # Create bar plot with error bars
        plt.clf()
        plt.figure(figsize=(10, 6))
        plt.bar(labels, means, yerr=stderrs, capsize=5, color='skyblue')
        plt.ylabel("score")
        plt.xticks(rotation=45, ha='right')
        if(self.num_folds == 1):
            plt.title(f"{metric.upper()} mean of {self.dev_stage}")
        else:
            plt.title(f"{metric.upper()} mean ± stderr of {self.dev_stage}")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{self.out_dir}/{metric}.svg", dpi=600)

    def rename_logs(self, parent_dir):
        for representation in self.representations:
            new_logs_dir = f"{self.base_dir}/{parent_dir}/{self.hyperparam}_{self.hyperparam_value}/{representation}/"
            if not os.path.exists(new_logs_dir):
                os.makedirs(new_logs_dir)
                os.rename(f"{self.base_dir}/{representation}/", 
                          f"{new_logs_dir}"
                )
    
    def run(self):
        mpl.rcParams['figure.dpi'] = 600
        mpl.rcParams.update({'font.size': 12})
        
        if not os.path.exists(self.out_dir): os.makedirs(self.out_dir)

        df = self.initialise_df()
        
        for metric_of_interest in ["mcc", "pr_auc", "roc_auc", "accuracy", "f1", "tn", "fp", "fn", "tp"]:
            results, epochs, metrics_arr, best_epochs_arr = self.get_metric_of_interest_values( metric_of_interest)
            df["best_epoch"] = best_epochs_arr
            df[metric_of_interest] = metrics_arr
            if(metric_of_interest not in ["tn", "fp", "fn", "tp"]):
                means, stderrs, labels = self.get_performance_stats(results)
                self.plot_stats(means, stderrs, labels, metric_of_interest)
        
        df.to_csv(f"{self.out_dir}/metrics.csv", index=False)
        
        