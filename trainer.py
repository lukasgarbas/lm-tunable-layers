import torch
import os
import math
import datetime
from copy import deepcopy
from pathlib import Path
from typing import Union
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from utils import Corpus
from task_heads import TextClassifier, TextRegressor

from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef
)


class ModelTrainer:
    def __init__(
        self,
        model: Union[TextClassifier, TextRegressor],
        corpus: Corpus,
        optimizer: torch.optim.Optimizer = AdamW,
    ):
        self.model = model
        self.corpus = corpus
        self.optimizer = optimizer
        self.evaluation_metrics = corpus.evaluation_metric
        self.main_metric = corpus.main_metric
        self.best_model = None
        self.use_linear_scheduler = False
        self.scheduler = None
        self.epoch = 0
        self.best_model_score = 0.0

        self.device = None
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.model.to(self.device)


    def train(
        self,
        learning_rate: float = 2e-5,
        batch_size: int = 16,
        epochs: int = 10,
        shuffle_data: bool = False,
        num_workers: int = 0,
        base_path: Union[Path, str] = None,
        save_best_model: bool = True,
        use_linear_scheduler: bool = False,
        warmup_fraction: float = 0.1,
        weight_decay: float = 0.01,
    ):
        self.optimizer = self.optimizer(self.model.parameters(),
                                        lr=learning_rate,
                                        weight_decay=weight_decay)

        num_updates_per_epoch = math.ceil(len(self.corpus.train)/batch_size)
        num_training_steps = math.ceil(epochs * num_updates_per_epoch)

        self.use_linear_scheduler = use_linear_scheduler
        if self.use_linear_scheduler:
            num_warmup_steps = math.ceil(num_training_steps*warmup_fraction)
            self.scheduler = self._create_linear_scheduler(num_training_steps,
                                                           num_warmup_steps)

        log_line(f"Model: {self.model}")
        log_line("----------------------------------------------------")
        log_line(f"Corpus:     {self.corpus.name}")
        log_line(f"Split:      {self.corpus.show_data_split()}")
        log_line(f"Task type:  {self.corpus.task_type}")
        log_line(f"Evaluation metric:  {', '.join(self.evaluation_metrics)}")
        log_line("----------------------------------------------------")
        log_line(f"Model type:")
        log_line(f" - {self.model.name}")
        log_line("----------------------------------------------------")
        log_line(f"Model parameters:")
        log_line(f" - device:          {self.device}")
        log_line(f" - epochs:          {epochs}")
        log_line(f" - batch size:      {batch_size}")
        log_line(f" - training steps:  {num_training_steps}")
        log_line(f" - optimizer:       {type(self.optimizer).__name__}")
        log_line(f" - learning rate:   {learning_rate}")
        if use_linear_scheduler:
            log_line(f" - linear schedule: {use_linear_scheduler}")
            log_line(f" - warmup fraction: {warmup_fraction}")
        log_line(f" - weight decay:    {weight_decay}")
        log_line(f" - shuffle data:    {shuffle_data}")
        log_line(f" - save best model: {save_best_model}")
        log_line("-----------------------------------------------------")

        # create base path where to save the model if not provided
        if not base_path:
            corpus_prefix = f"{self.corpus.name[:4].strip('- ')}-" if self.corpus.name else ""
            base_path = f"models/{corpus_prefix}{self.model.name}"

        if type(base_path) is str:
            base_path = Path(base_path)

        # create batches for dev data
        dev_batch_loader = self.corpus.create_dev_dataloader(
            batch_size=batch_size,
            num_workers=num_workers,
        )

        train_loss = []
        dev_loss = []
        dev_score_history = []

        self.epoch = 0

        self.model.train()

        for epoch in range(epochs):

            # keep track of epochs for logging purposes
            self.epoch = epoch + 1

            # shuffle the data at each epoch except the first one
            # shuffle is false by default
            train_batch_loader = self.corpus.create_train_dataloader(
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle_data if self.epoch > 1 else False,
            )

            # training step
            train_epoch_loss = self.fit(train_batch_loader)
            train_loss.append(train_epoch_loss)

            # validation step
            dev_epoch_loss, metrics = self.validate(dev_batch_loader)
            dev_loss.append(dev_epoch_loss)
            dev_score_history.append(metrics[self.main_metric])

            log_line("-----------------------------------------------------")
            log_line(f"EPOCH {epoch+1} done: train loss avg {train_epoch_loss:.4f}")
            log_line(f"DEV: loss {dev_epoch_loss:.4f} {metrics_log(metrics)}")

            # save model with the highest score
            current_score = metrics[self.main_metric]
            if current_score > self.best_model_score:
                self.best_model_score = current_score
                self.best_model = deepcopy(self.model)

                # store trained models using torch.save()
                if save_best_model:
                    log_line("saving best model")
                    if not os.path.isdir(base_path):
                        os.makedirs(base_path)
                    torch.save(self.model, base_path / "best-model.pt")

            log_line("----------------------------------------------------")

        log_line("DONE TRAINING.")
        log_line(f"Best model score on dev set: {self.main_metric} {self.best_model_score}")

        # final evaluation of the model if corpus has a test set
        if self.corpus.test:
            log_line("----------------------------------------------------")
            log_line("Evaluating on test set...")
            test_batches = self.corpus.create_test_dataloader(batch_size=batch_size,
                                                              num_workers=num_workers)
            _, metrics = self.validate(test_batches, use_best_model=True)
            log_line(f"Corpus: {self.corpus.name}")
            log_line(f"Task type: {self.corpus.task_type}")
            log_line(f"TEST set performance: {metrics_log(metrics)}")
            log_line("----------------------------------------------------")

        history = {
            "train_loss": train_loss,
            "val_loss": dev_loss,
            "dev_score_history": dev_score_history
        }
        return history


    def fit(self, train_batches):
        self.model.train()
        train_loss = 0.0
        intermittent_loss = 0.0

        seen_batches = 0
        number_of_batches = len(train_batches)
        modulo = max(1, int(number_of_batches / 10))

        for data, targets in train_batches:

            data = list(data)
            targets = targets.to(self.device)

            # zero the gradients on the model and optimizer
            self.model.zero_grad()
            self.optimizer.zero_grad()

            # forward pass
            loss = self.model.forward_loss(data, targets)

            loss_item = loss.item()
            train_loss += loss_item
            intermittent_loss += loss_item

            seen_batches += 1

            loss.backward()

            if seen_batches % modulo == 0:

                # display current learning rate
                learning_rate_info = "lr - "
                for group in self.optimizer.param_groups:
                    learning_rate_info += f"{group['lr']:.0e} "

                intermittent_loss = intermittent_loss / modulo
                log_line(f"epoch {self.epoch} - iter {seen_batches}/{number_of_batches}"
                        f" - loss {intermittent_loss:.5f} - {learning_rate_info}")
  
            # perform optimizer step
            self.optimizer.step()

            # scheduler step: change the learning rate using linear decay scheduler
            if self.use_linear_scheduler:
                self.scheduler.step()

        avg_train_loss = float(train_loss / number_of_batches)
        return avg_train_loss


    def validate(self, batches, use_best_model=False):
        model = self.best_model if use_best_model else self.model

        # sets the module in evaluation mode
        model.eval()

        # sum up the losses after each forward pass
        validation_loss = 0.0
        predictions = []
        ground_truths = []

        for data, targets in batches:
            data = list(data)
            targets = targets.to(self.device)

            with torch.no_grad():
                loss = model.forward_loss(data, targets)

            validation_loss += loss.item()

            prediction = model.predict(data)
            predictions.extend(prediction.cpu())
            ground_truths.extend(targets.cpu())

        # validation loss average
        # val loss: sum of all batch losses / number of batches
        val_loss = float(validation_loss/len(batches))

        # get the evaluation score
        metrics = self._compute_score(predictions,
                                      ground_truths)
        return val_loss, metrics

    def _compute_score(self, predictions, ground_truths):
        metrics = {}
        # one model can be evaluated using multiple metrics
        for metric in self.evaluation_metrics:
            if metric == "accuracy":
                metrics["accuracy"] = accuracy_score(ground_truths, predictions)
            if metric == "f1_score":
                metrics["f1_score"] = f1_score(ground_truths, predictions, average="micro")
            if metric == "matthews_corr":
                metrics["matthews_corr"] = matthews_corrcoef(ground_truths, predictions)
            if metric == "pearsonr":
                metrics["pearsonr"] = pearsonr(ground_truths, predictions)[0]
            if metric == "spearmanr":
                metrics["spearmanr"] = spearmanr(ground_truths, predictions)[0]

        return metrics


    def _create_linear_scheduler(self, num_training_steps, num_warmup_steps, last_epoch=-1):
        """
        Linear scheduler with warmup: 
        increase the learning rate from zero to the given lr during warmup, 
        decrease the learning during from the given lr to zero for the rest training steps.
        """
        def lr_lambda(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
            )
        return LambdaLR(self.optimizer, lr_lambda, last_epoch)


def log_line(text_line):
    dt = datetime.datetime.now().isoformat(" ", "seconds")
    log_string = f"{dt} {text_line}"
    print(log_string)


def metrics_log(metrics):
    metric_results = ""
    for metric, score in metrics.items():
        metric_results += f"{str(metric)} {str(round(score, 4))} "
    return metric_results

