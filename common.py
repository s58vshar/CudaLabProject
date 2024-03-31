import numpy as np
import torch
import os
import math

def save_model(model, optimizer, epoch, stats, metrics):
    """Saving model checkpoint along with metrics."""
    create_dir("checkpoints")
    savepath = f"checkpoints/checkpoint_epoch_{epoch}.pth"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats,
        'metrics': metrics  # Include metrics here
    }, savepath)


def load_model(model, optimizer, savepath):
    """Loading pretrained checkpoint along with metrics."""
    
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint.get("stats", None)  # Use .get to avoid KeyError if not found
    metrics = checkpoint.get("metrics", None)  # Extract metrics

    return model, optimizer, epoch, stats, metrics

def create_dir(path):
    """ Creating directory if it does not exist already """
    if not os.path.exists(path):
        os.makedirs(path)
    return



def IoU(pred, target, num_classes):
    """ Computing the IoU for a single image """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for lbl in range(num_classes):
        pred_inds = pred == lbl
        target_inds = target == lbl
        
        intersection = (pred_inds[target_inds]).long().sum().cpu()
        union = pred_inds.long().sum().cpu() + target_inds.long().sum().cpu() - intersection
        iou = intersection / (union + 1e-8)
        iou = iou + 1e-8 if union > 1e-8 and not math.isnan(iou) else 0
        ious.append(iou)
    return torch.tensor(ious)



def calculate_mean_accuracy(pred, target, num_classes):
    """Calculate mean accuracy across all classes for a single image or batch."""
    correct_per_class = torch.zeros(num_classes)
    total_per_class = torch.zeros(num_classes)

    for cls in range(num_classes):
        correct_per_class[cls] = (pred[target == cls] == cls).sum()
        total_per_class[cls] = (target == cls).sum()

    # Avoid division by zero and only consider classes present in the target
    mean_accuracy = correct_per_class[total_per_class > 0] / total_per_class[total_per_class > 0]
    mean_accuracy = mean_accuracy.mean()  # Average over all present classes
    return mean_accuracy


