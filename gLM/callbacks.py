import torch
from pytorch_lightning.callbacks import Callback
from sklearn import metrics


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
        y_pred = torch.cat(pl_module.step_outputs[split][self.y_pred_key], dim=0)
        y_true = torch.cat(pl_module.step_outputs[split][self.y_true_key], dim=0)
        log_dict = log_classification_metrics(
            y_pred,
            y_true,
            y_true_key=self.y_true_key,
            y_pred_key=self.y_pred_key,
            prefix=split,
            class_names=self.class_names,
            make_one_hot=self.make_one_hot,
            invert_probabilities=self.invert_probabilities,
            make_correct_dim=self.make_correct_dim,
            metrics_to_plot=self.metrics_to_plot,
        )
        trainer.logger.experiment.log(log_dict)

    def on_validation_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "val")

    def on_train_epoch_end(self, trainer, pl_module):
        self.log_metrics(trainer, pl_module, "train")
