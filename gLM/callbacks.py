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

def log_logit_classification_metrics(
    y_pred_lab,
    y_true_lab,
    prefix,
    trainer,
    metrics_to_plot=[
        "mcc",
        "confusion_matrix",
        "class_proportions",
        "classification_report",
    ],
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

    if "roc" in metrics_to_plot:
        log_dict[f"{prefix}_roc"] = metrics.roc_auc_score(y_true_lab, y_pred)

    if "confusion_matrix" in metrics_to_plot:
        log_dict[f"{prefix}_confusion_matrix"] = wandb.plot.confusion_matrix(probs=None,
            y_true=y_true_lab, preds=y_pred_lab,
            class_names=class_names)

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


def log_classification_metrics(
    y_pred,
    y_true,
    y_true_key,
    y_pred_key,
    prefix,
    trainer,
    metrics_to_plot=[
        "confusion_matrix",
        "roc",
        "pr",
        "class_proportions",
        "calibration_curves",
        "classification_report",
    ],
    class_names=["Negative", "Positive"],
    make_one_hot=False,
    invert_probabilities=False,
    make_correct_dim=False,
):
    if invert_probabilities:
        y_pred = 1 - y_pred
    if make_one_hot:
        y_true = F.one_hot(y_true, num_classes=len(class_names))
    if make_correct_dim:
        y_pred = torch.stack([1 - y_pred, y_pred], dim=1)
    y_pred_class = torch.argmax(y_pred, dim=1)
    y_true_class = torch.argmax(y_true, dim=1).detach().cpu().numpy().astype(int)

    y_pred = y_pred.cpu().detach()
    y_pred_class = y_pred_class.cpu().detach()

    log_dict = {}
    suffix_title = f"{y_true_key} vs {y_pred_key}"
    if "confusion_matrix" in metrics_to_plot:
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_confusion_matrix"] = wandb.plot.confusion_matrix(
            probs=y_pred,
            y_true=y_true_class,
            class_names=class_names,
            split_table=True,
            title=f"{suffix_title} Confusion Matrix",
        )

    if "roc" in metrics_to_plot:
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_roc_auc"] = metrics.roc_auc_score(
            y_true_class, y_pred[:, 1]
        )

    if "roc_curve" in metrics_to_plot:
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_roc"] = wandb.plot.roc_curve(
            y_true_class,
            y_pred,
            labels=class_names,
            split_table=True,
            title=f"{suffix_title} ROC Curve",
        )

    if "pr" in metrics_to_plot:
        # log the average precision value
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_average_precision"] = (
            metrics.average_precision_score(y_true_class, y_pred[:, 1])
        )

    if "pr_curve" in metrics_to_plot:
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_pr"] = wandb.plot.pr_curve(
            y_true_class,
            y_pred,
            labels=class_names,
            split_table=True,
            title=f"{suffix_title} PR Curve",
        )

    if "class_proportions" in metrics_to_plot:
        num_classes = len(class_names)
        class_counts = np.bincount(y_true_class, minlength=num_classes)
        class_proportions = class_counts / len(y_true_class)
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_class_proportions"] = wandb.plot.bar(
            wandb.Table(
                data=[
                    [class_names[i], class_proportions[i]] for i in range(num_classes)
                ],
                columns=["Class", "Proportion"],
            ),
            "Class",
            "Proportion",
            title=f"{suffix_title} Class Proportions",
            split_table=True,
        )

    if "calibration_curves" in metrics_to_plot:
        y_pred_np = np.minimum(np.maximum(y_pred.numpy(), 0), 1)
        diagonal = np.linspace(0, 1, num=100)
        xs, ys, keys = [diagonal], [diagonal], ["Perfect calibration"]
        for i in range(num_classes):
            prob_true, prob_pred = calibration_curve(
                y_true_class == i, y_pred_np[:, i], n_bins=10
            )
            xs.append(prob_pred)
            ys.append(prob_true)
            keys.append(class_names[i])
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_calibration_curves"] = wandb.plot.line_series(
            xs=xs,
            ys=ys,
            keys=keys,
            title=f"{suffix_title} Calibration Curves",
            xname="Mean predicted probability",
            split_table=True,
        )

    if "classification_report" in metrics_to_plot:
        class_report = metrics.classification_report(
            y_true_class, y_pred_class, target_names=class_names
        ).split("\n")
        report_columns = ["Class", "Precision", "Recall", "F1-score", "Support"]
        report_table = []
        for line in class_report[2 : (num_classes + 2)]:
            report_table.append(line.split())
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_classification_report"] = wandb.Table(
            dataframe=pd.DataFrame(report_table, columns=report_columns)
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
        predicted_labels = pl_module.step_outputs[split]['predicted_label']
        predictions = pl_module.step_outputs[split]['predictions']

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
        pl_module.step_outputs["val"].clear()

    def on_validation_epoch_end(self, trainer, pl_module):
        self.on_epoch_end(trainer, pl_module, 'val')

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

        log_dict = log_logit_classification_metrics(
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
        self.log_metrics(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "test")



