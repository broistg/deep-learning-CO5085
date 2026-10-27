"""E1 training and evaluation. `train_one_epoch`, `evaluate_one_epoch`, `count_parameters` and `predict` are Ly's notebook code,
moved unchanged. `train_model` has three changes (marked CHANGED): it times the epochs with a GPU sync, it keeps the FINAL weights in
`model` (the main number is the last epoch, like E2/A2) and returns the best-validation-loss checkpoint separately instead of
loading it and writing a file into the current folder."""
import copy
import time

import numpy as np
import torch
import torch.nn as nn


def evaluate_one_epoch(model, dataloader, criterion, device):
  model.eval()

  total_correct = 0
  total_samples = 0
  total_loss = 0

  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)

      logits = model(images)
      loss = criterion(logits, labels)

      predictions = logits.argmax(dim=1)

      total_correct += (predictions == labels).sum().item()
      total_samples += labels.size(0)
      total_loss += loss.item() * images.size(0)

  average_loss = total_loss / total_samples
  accuracy = total_correct / total_samples

  return {
      "loss": average_loss,
      "accuracy": accuracy
  }


def train_one_epoch(model, dataloader, criterion, optimizer, device):
  model.train()
  total_correct = 0
  total_samples = 0
  total_loss = 0

  for images, labels in dataloader:
    images, labels = images.to(device), labels.to(device)
    optimizer.zero_grad()

    logits = model(images)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()

    total_loss += loss.item() * images.size(0)
    predictions = logits.argmax(dim=1)
    total_correct += (predictions == labels).sum().item()
    total_samples += labels.size(0)

  return {
      "loss": total_loss / total_samples,
      "accuracy": total_correct / total_samples
  }


def count_parameters(model):
  total = sum(p.numel() for p in model.parameters())
  trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
  return total, trainable


def predict(model, dataloader, device):
  model.eval()

  y_true = []
  y_pred = []
  y_prob = []

  start_time = time.time()
  with torch.no_grad():
    for images, labels in dataloader:
      images, labels = images.to(device), labels.to(device)
      logits = model(images)
      predictions = logits.argmax(dim=1)
      probabilities = torch.softmax(logits, dim=1)

      # Store batch tensors directly
      y_true.append(labels.cpu())
      y_pred.append(predictions.cpu())
      y_prob.append(probabilities.cpu())
  end_time = time.time()


  # Concatenate along the first dimension (batch dimension)
  y_true = torch.cat(y_true, dim=0).numpy()
  y_pred = torch.cat(y_pred, dim=0).numpy()
  y_prob = torch.cat(y_prob, dim=0).numpy()

  return y_true, y_pred, y_prob, end_time - start_time


def train_model(model, train_loader, val_loader, device, epochs=3, lr=0.001, log=print):
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "epoch_times": [],
        "val_times": []
    }

    best = {"val_loss": float("inf"), "state": None, "epoch": 0}
    if device.type == "cuda":                                           # CHANGED: peak memory for the cost table
        torch.cuda.reset_peak_memory_stats()

    for epoch in range(epochs):
        # Training
        if device.type == "cuda":                                       # CHANGED: GPU work is asynchronous, sync before reading the clock
            torch.cuda.synchronize()
        start_time = time.time()
        train_result = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        end_time = time.time()

        val_result = evaluate_one_epoch(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device
        )

        # Append metrics to history
        history["train_losses"].append(train_result["loss"])
        history["train_accuracies"].append(train_result["accuracy"])
        history["val_losses"].append(val_result["loss"])
        history["val_accuracies"].append(val_result["accuracy"])
        history["epoch_times"].append(end_time - start_time)
        history["val_times"].append(time.time() - end_time)

        if val_result["loss"] < best["val_loss"]:
            best = {"val_loss": val_result["loss"], "state": copy.deepcopy(model.state_dict()), "epoch": epoch + 1}

        log(f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_result['loss']:.4f} | Train Acc: {train_result['accuracy']:.4f} | "
            f"Val Loss: {val_result['loss']:.4f} | Val Acc: {val_result['accuracy']:.4f}")

    # CHANGED: `model` keeps the weights of the LAST epoch; the best-val-loss checkpoint is returned in `best`
    total, trainable = count_parameters(model)
    history.update(best_epoch=best["epoch"], params_total=total, params_trainable=trainable,
                   peak_mem_mb=torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None)
    return model, history, best
