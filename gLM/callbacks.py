import torch
from pytorch_lightning.callbacks import Callback
from sklearn import metrics
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import wandb

class SetupWandB(Callback):
    def on_train_start(self, trainer, pl_module):
        wandb.watch(pl_module.model, log="all", log_graph=True, log_freq=1)

def log_classification_metrics(
    y_pred_lab,
    y_true_lab,
    prefix,
    trainer,
    metrics_to_plot=(
        "mcc",
        "confusion_matrix",
        "class_proportions",
        "classification_report",
    ),
    class_names=["Negative", "Positive"],
    y_pred=None
):
    log_dict = {}

    y_true_lab = y_true_lab.detach().cpu().numpy().astype(int)
    y_pred_lab = y_pred_lab.detach().cpu().numpy().astype(int)
    y_pred = y_pred.detach().cpu().numpy().astype(int)

    num_classes = len(class_names)

    if "mcc" in metrics_to_plot:
        log_dict[f"{prefix}_mcc"] = metrics.matthews_corrcoef(y_true_lab, y_pred_lab)

    if "roc" in metrics_to_plot or "roc_auc" in metrics_to_plot:
        log_dict[f"{prefix}_roc_auc"] = metrics.roc_auc_score(y_true_lab, y_pred)

    if "pr_auc" in metrics_to_plot:
        precision, recall, _ = precision_recall_curve(y_true_lab, y_pred)
        pr_auc = auc(recall, precision)

        log_dict[f"{prefix}_pr_auc"] = pr_auc

    if "confusion_matrix" in metrics_to_plot:
        cm = confusion_matrix(y_true_lab, y_pred_lab)
        # Extract TN, FP, FN, TP
        tn, fp, fn, tp = cm.ravel()
        
        # Create a labeled confusion matrix
        labeled_cm = np.array([['TN', tn], ['FP', fp], ['FN', fn], ['TP', tp]])

        cm_df = pd.DataFrame(labeled_cm)
        log_dict[f"{prefix}_confusion_matrix"] = wandb.Table(dataframe=cm_df)

    if "classification_report" in metrics_to_plot:
        class_report = metrics.classification_report(
            y_true_lab, y_pred_lab, target_names=class_names
        ).split("\n")
        report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
        report_table = []
        for line in class_report[2 : (num_classes + 2)]:
            report_table.append(line.split())
        log_dict[f"{prefix}_classification_report"] = wandb.Table(
            dataframe=pd.DataFrame(report_table, columns=report_columns)
        )

    if "class_proportions" in metrics_to_plot:
        class_counts = np.bincount(y_true_lab, minlength=num_classes)
        class_proportions = class_counts / len(y_true_lab)
        log_dict[f"{prefix}_class_proportions"] = wandb.plot.bar(
            wandb.Table(
                data=[
                    [class_names[i], class_proportions[i]] for i in range(num_classes)
                ],
                columns=["Class", "Proportion"],
            ),
            "Class",
            "Proportion",
            title=f"Class Proportions",
            split_table=True,
        )
        
    return log_dict

class OutputLoggingCallback(Callback):
    def __init__(self, log_every_n_steps=20):
        self.log_every_n_steps = log_every_n_steps
        self.last_logged_step = 0

    def on_epoch_end(self, trainer, pl_module, split):
        # Create a table with protein IDs, true labels, and predicted labels
        protein_ids = pl_module.step_outputs[split]['concat_id']
        true_labels = pl_module.step_outputs[split]['label']
        predicted_labels = pl_module.step_outputs[split]['predicted_label'].squeeze().numpy()
        predictions = pl_module.step_outputs[split]['predictions'].squeeze().numpy()

        table_data = []
        for pid, true, pred_lab, pred in zip(protein_ids, true_labels, predicted_labels, predictions):
            table_data.append([pid, true, pred_lab, pred])

        table = wandb.Table(data=table_data, columns=["Complex ID", "True label", "Predicted label", "Prediction"])
        trainer.logger.experiment.log({f"{split}_epoch_end": table})

    def on_fit_start(self, trainer, pl_module):
        # Initialize step_outputs in the pl_module if it doesn't exist
        if not hasattr(pl_module, "step_outputs"):
            pl_module.step_outputs = defaultdict(lambda: defaultdict(list))

    def on_test_start(self, trainer, pl_module):
        # Initialize step_outputs in the pl_module if it doesn't exist
        if not hasattr(pl_module, "step_outputs"):
            pl_module.step_outputs = defaultdict(lambda: defaultdict(list))
            
    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.step_outputs["train"].clear()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.step_outputs["validate"].clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, 'validate')

    def on_test_epoch_start(self, trainer, pl_module):
        pl_module.step_outputs["test"].clear()

    def on_test_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, 'test')


class LogClassificationMetrics(Callback):
    def __init__(
        self,
        class_names: list,
        y_pred_key: str,
        y_pred_lab_key: str,
        y_true_lab_key: str,
        make_one_hot: bool = False,
        invert_probabilities: bool = False,
        make_correct_dim: bool = False,
        metrics_to_plot: list = [
            "mcc",
            "confusion_matrix",
            "class_proportions",
            "classification_report",
        ],
    ):
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.y_pred_key = y_pred_key
        self.y_pred_lab_key = y_pred_lab_key
        self.y_true_lab_key = y_true_lab_key
        self.make_one_hot = make_one_hot
        self.invert_probabilities = invert_probabilities
        self.make_correct_dim = make_correct_dim
        if self.make_correct_dim:
            assert (
                self.num_classes == 2
            ), "make_correct_dim only works for binary classification"
        self.metrics_to_plot = metrics_to_plot

    def log_metrics(self, trainer, pl_module, split):
        y_pred = torch.cat([pl_module.step_outputs[split][self.y_pred_key]], dim=0)
        y_pred_lab = torch.cat([pl_module.step_outputs[split][self.y_pred_lab_key]], dim=0)
        y_true_lab = torch.cat([pl_module.step_outputs[split][self.y_true_lab_key]], dim=0)

        log_dict = log_classification_metrics(
            y_pred_lab,
            y_true_lab,
            y_pred=y_pred,
            prefix=split,
            trainer=trainer,
            class_names=self.class_names,
            metrics_to_plot=self.metrics_to_plot,
        )
        
        trainer.logger.experiment.log(log_dict)
        
    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "validate")

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "test")



