import torch
from pytorch_lightning.callbacks import Callback
from sklearn import metrics
import pandas as pd
import torch.nn.functional as F
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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
    y_true_class = torch.argmax(y_true, dim=1).numpy().astype(int)

    log_dict = {}
    suffix_title = f"{y_true_key} vs {y_pred_key}"
    if "confusion_matrix" in metrics_to_plot:
        cm = confusion_matrix(y_true_class, y_pred_class)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f"{suffix_title} Confusion Matrix")
        plt.savefig(f"{prefix}_{y_pred_key}_{y_true_key}_confusion_matrix.png")
        plt.close()
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, f"{prefix}_{y_pred_key}_{y_true_key}_confusion_matrix.png")

    if "roc" in metrics_to_plot:
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_roc_auc"] = metrics.roc_auc_score(
            y_true_class, y_pred[:, 1]
        )

    if "roc_curve" in metrics_to_plot:
        fpr, tpr, _ = roc_curve(y_true_class, y_pred[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f"{suffix_title} ROC Curve")
        plt.legend(loc="lower right")
        plt.savefig(f"{prefix}_{y_pred_key}_{y_true_key}_roc_curve.png")
        plt.close()
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, f"{prefix}_{y_pred_key}_{y_true_key}_roc_curve.png")

    if "pr" in metrics_to_plot:
        # log the average precision value
        log_dict[f"{prefix}_{y_pred_key}/{y_true_key}_average_precision"] = (
            metrics.average_precision_score(y_true_class, y_pred[:, 1])
        )

    if "pr_curve" in metrics_to_plot:
        precision, recall, _ = precision_recall_curve(y_true_class, y_pred[:, 1])
        pr_auc = auc(recall, precision)
        plt.figure()
        plt.plot(recall, precision, color='b', lw=2, label=f'PR curve (area = {pr_auc:0.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f"{suffix_title} PR Curve")
        plt.legend(loc="lower left")
        plt.savefig(f"{prefix}_{y_pred_key}_{y_true_key}_pr_curve.png")
        plt.close()
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, f"{prefix}_{y_pred_key}_{y_true_key}_pr_curve.png")

    if "class_proportions" in metrics_to_plot:
        num_classes = len(class_names)
        class_counts = np.bincount(y_true_class, minlength=num_classes)
        class_proportions = class_counts / len(y_true_class)
        plt.figure(figsize=(10, 7))
        plt.bar(class_names, class_proportions)
        plt.xlabel('Class')
        plt.ylabel('Proportion')
        plt.title(f"{suffix_title} Class Proportions")
        plt.savefig(f"{prefix}_{y_pred_key}_{y_true_key}_class_proportions.png")
        plt.close()
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, f"{prefix}_{y_pred_key}_{y_true_key}_class_proportions.png")

    if "calibration_curves" in metrics_to_plot:
        y_pred_np = np.minimum(np.maximum(y_pred.numpy(), 0), 1)
        diagonal = np.linspace(0, 1, num=100)
        plt.figure(figsize=(10, 7))
        plt.plot(diagonal, diagonal, linestyle='--', color='gray', label='Perfect calibration')
        for i in range(len(class_names)):
            prob_true, prob_pred = calibration_curve(y_true_class == i, y_pred_np[:, i], n_bins=10)
            plt.plot(prob_pred, prob_true, marker='o', label=class_names[i])
        plt.xlabel('Mean predicted probability')
        plt.ylabel('Fraction of positives')
        plt.title(f"{suffix_title} Calibration Curves")
        plt.legend()
        plt.savefig(f"{prefix}_{y_pred_key}_{y_true_key}_calibration_curves.png")
        plt.close()
        trainer.logger.experiment.log_artifact(trainer.logger.run_id, f"{prefix}_{y_pred_key}_{y_true_key}_calibration_curves.png")

    if "classification_report" in metrics_to_plot:
        class_report = metrics.classification_report(
            y_true_class, y_pred_class, target_names=class_names, output_dict=True
        )
        for class_name, metrics_dict in class_report.items():
            if isinstance(metrics_dict, dict):
                for metric_name, value in metrics_dict.items():
                    log_dict[f"{prefix}_{y_pred_key}_{y_true_key}_{class_name}_{metric_name}"] = value
    return log_dict

class OutputLoggingCallback(Callback):
    def __init__(self, log_every_n_steps=20):
        self.log_every_n_steps = log_every_n_steps
        self.last_logged_step = 0

    def on_fit_start(self, trainer, pl_module):
        # Initialize step_outputs in the pl_module if it doesn't exist
        if not hasattr(pl_module, "step_outputs"):
            pl_module.step_outputs = defaultdict(lambda: defaultdict(list))
        mlflow.set_experiment(trainer.logger._experiment_name)
        mlflow.start_run(run_name=f"{trainer.logger._run_name}_system_metrics", log_system_metrics=True)

    def _log_outputs(self, pl_module, batch, split):
        output_dict = pl_module.get_log_outputs(batch)

        for key in output_dict:
            if not isinstance(output_dict[key], torch.Tensor):
                continue
            pl_module.step_outputs[split][key].append(output_dict[key].cpu().detach())

    def on_train_epoch_start(self, trainer, pl_module):
        pl_module.step_outputs["train"].clear()

    def on_validation_epoch_start(self, trainer, pl_module):
        pl_module.step_outputs["val"].clear()


class LogClassificationMetrics(Callback):
    def __init__(
        self,
        class_names: list,
        y_pred_key: str,
        y_true_key: str,
        make_one_hot: bool = False,
        invert_probabilities: bool = False,
        make_correct_dim: bool = False,
        metrics_to_plot: list = [
            "confusion_matrix",
            "roc",
            "roc_curve",
            "pr",
            "pr_curve",
            "class_proportions",
            "calibration_curves",
            "classification_report",
        ],
    ):
        self.num_classes = len(class_names)
        self.class_names = class_names
        self.y_pred_key = y_pred_key
        self.y_true_key = y_true_key
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
        y_true = torch.cat([pl_module.step_outputs[split][self.y_true_key]], dim=0)

        log_dict = log_classification_metrics(
            y_pred,
            y_true,
            y_true_key=self.y_true_key,
            y_pred_key=self.y_pred_key,
            prefix=split,
            trainer=trainer,
            class_names=self.class_names,
            make_one_hot=self.make_one_hot,
            invert_probabilities=self.invert_probabilities,
            make_correct_dim=self.make_correct_dim,
            metrics_to_plot=self.metrics_to_plot,
        )
        
        for key, value in log_dict.items():
            trainer.logger.experiment.log_metric(trainer.logger.run_id, key=key, value=value)
        

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "val")

    def on_test_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "test")



