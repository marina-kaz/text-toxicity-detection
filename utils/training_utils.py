import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from utils.parameters import LEARNING_RATE, WEIGHT_DECAY, OPTIMIZER_BETAS, CORRECT_BIAS
import transformers
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data.dataloader import DataLoader
from typing import Tuple, Union


def train_epoch(
    train_dataloader: DataLoader,
    model: transformers.PreTrainedModel,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: _LRScheduler,
    device: Union[str, torch.device]
) -> Tuple[float, float]:
    """
    Train the model for one epoch.

    Args:
        train_dataloader (DataLoader): DataLoader for training data.
        model (transformers.PreTrainedModel): Model to be trained.
        criterion (torch.nn.Module): Loss criterion.
        optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
        scheduler (_LRScheduler): Learning rate scheduler.
        device (Union[str, torch.device]): Device to perform training.

    Returns:
        Tuple[float, float]: Tuple containing average loss and accuracy for the epoch.
    """
    model.train()
    correct_predictions = 0
    all_predictions = 0
    losses = []

    for batch in train_dataloader:
        optimizer.zero_grad()

        ids = batch["ids"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        output = model(ids, mask)["logits"].squeeze(-1).to(torch.float32)
        preds = torch.where(torch.sigmoid(output) > 0.5, 1, 0)

        toxic_label = batch["toxic_label"].to(device, non_blocking=True)
        loss = criterion(output, toxic_label)
        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        correct_predictions += torch.sum(preds == toxic_label)
        all_predictions += len(preds)

    losses = np.mean(losses)
    accuracy = correct_predictions.detach().cpu().numpy() / all_predictions

    return losses, accuracy


def validate(
    valid_dataloader: DataLoader,
    model: transformers.PreTrainedModel,
    device: Union[str, torch.device],
    criterion: torch.nn.Module
) -> Tuple[float, float]:
    """
    Validate the model on validation data.

    Args:
        valid_dataloader (DataLoader): DataLoader for validation data.
        model (transformers.PreTrainedModel): Model to be validated.
        device (Union[str, torch.device]): Device to perform validation.
        criterion (torch.nn.Module): Loss criterion.

    Returns:
        Tuple[float, float]: Tuple containing average loss and accuracy on validation data.
    """
    model.eval()
    correct_predictions = 0
    all_predictions = 0
    losses = []

    with torch.no_grad():
        for batch in valid_dataloader:
            ids = batch["ids"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            output = model(ids, mask)["logits"].squeeze(-1).to(torch.float32)
            preds = torch.where(torch.sigmoid(output) > 0.5, 1, 0)

            toxic_label = batch["toxic_label"].to(device, non_blocking=True)
            loss = criterion(output, toxic_label)
            losses.append(loss.item())

            correct_predictions += torch.sum(preds == toxic_label)
            all_predictions += len(preds)

    losses = np.mean(losses)
    accuracy = correct_predictions / all_predictions

    return losses, accuracy


class Trainer:
    """
    Class for training and validating a model.
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        criterion: torch.nn.Module,
        device: Union[str, torch.device],
        ckpt_path: Path
    ) -> None:
        """
        Initialize the Trainer.

        Args:
            model (transformers.PreTrainedModel): Model to be trained.
            criterion (torch.nn.Module): Loss criterion.
            device (Union[str, torch.device]): Device for training.
            ckpt_path (Path): Path to save model checkpoints.
        """
        self._model = model
        self._criterion = criterion
        self._device = device
        self._ckpt_path = ckpt_path

    def train(
        self,
        train_dataloader: DataLoader,
        valid_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: _LRScheduler,
        epochs: int
    ) -> transformers.PreTrainedModel:
        """
        Train the model.

        Args:
            train_dataloader (DataLoader): DataLoader for training data.
            valid_dataloader (DataLoader): DataLoader for validation data.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            scheduler (_LRScheduler): Learning rate scheduler.
            epochs (int): Number of epochs to train.

        Returns:
            transformers.PreTrainedModel: Trained model.
        """
        best_score = None

        for _ in tqdm(range(epochs)):
            train_loss, train_acc = train_epoch(
                train_dataloader=train_dataloader,
                model=self._model,
                criterion=self._criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                device=self._device,
            )
            valid_loss, valid_acc = validate(valid_dataloader, self._model, self._device, self._criterion)

            print("train loss: %.4f" % train_loss, "train accuracy: %.3f" % train_acc)
            print("valid loss: %.4f" % valid_loss, "valid accuracy: %.3f" % valid_acc)

            if best_score is None:
                best_score = valid_loss
            elif valid_loss < best_score:
                best_score = valid_loss
                state = {
                    "state_dict": self._model.state_dict(),
                    "optimizer_dict": optimizer.state_dict(),
                    "best_score": best_score,
                }
                torch.save(state, self._ckpt_path)

        return self._model


def get_optimizer(model_parameters) -> torch.optim.Optimizer:
    """
    Get optimizer for training.

    Args:
        model_parameters: Parameters of the model.

    Returns:
        torch.optim.Optimizer: Optimizer for training.
    """
    return transformers.AdamW(
        model_parameters,
        LEARNING_RATE,
        betas=OPTIMIZER_BETAS,
        weight_decay=WEIGHT_DECAY,
        correct_bias=CORRECT_BIAS
    )
