"""
Utils methods for data visualization
"""

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib import colors
from torchvision.utils import draw_segmentation_masks


COLORS = ["blue", "green", "olive", "red", "yellow", "purple", "orange", "cyan",
          "brown", "pink", "darkorange", "goldenrod", "forestgreen", "springgreen",
          "aqua", "royalblue", "navy", "darkviolet", "plum", "magenta", "slategray",
          "maroon", "gold", "peachpuff", "silver", "aquamarine", "indianred", "greenyellow",
          "darkcyan", "sandybrown"]

VOC_COLORMAP = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 128, 0),
    (128, 128, 0),
    (0, 0, 128),
    (128, 0, 128),
    (0, 128, 128),
    (128, 128, 128),
    (64, 0, 0),
    (192, 0, 0),
    (64, 128, 0),
    (192, 128, 0),
    (64, 0, 128),
    (192, 0, 128),
    (64, 128, 128),
    (192, 128, 128),
    (0, 64, 0),
    (128, 64, 0),
    (0, 192, 0),
    (128, 192, 0),
    (0, 64, 128),
    (0,128,192),
    (128,192,192),
]


def visualize_sequence(sequence, savepath=None, add_title=True, add_axis=False, n_cols=10,
                       size=3, n_channels=3, titles=None, unnorm=False, **kwargs):
    """
    Visualizing a sequence of imgs in a grid like manner.

    Args:
    -----
    sequence: torch Tensor
        Sequence of images to visualize. Shape in (N_imgs, C, H, W)
    savepath: string ir None
        If not None, path where to store the sequence
    add_title: bool
        whether to add a title to each image
    n_cols: int
        Number of images per row in the grid
    size: int
        Size of each image in inches
    n_channels: int
        Number of channels (RGB=3, grayscale=1) in the data
    titles: list
        Titles to add to each image if 'add_title' is True
    """
    # initializing grid
    n_frames = sequence.shape[0]
    n_rows = int(np.ceil(n_frames / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols)

    # adding super-title and resizing
    figsize = kwargs.pop("figsize", (3*n_cols, 3*n_rows))
    fig.set_size_inches(*figsize)
    fig.suptitle(kwargs.pop("suptitle", ""))

    if unnorm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        sequence = sequence * std + mean

    # plotting all frames from the sequence
    ims = []
    for i in range(n_frames):
        row, col = i // n_cols, i % n_cols
        a = ax[row, col] if n_rows > 1 else ax[col]
        if(sequence[i].dim()==2) :
            f = sequence[i].cpu().detach()
        else :
            f = sequence[i].permute(1, 2, 0).cpu().detach()
        if(n_channels == 1):
            f = f[..., 0]
        im = a.imshow(f, **kwargs)
        ims.append(im)
        if(add_title):
            if(titles is not None):
                cur_title = "" if i >= len(titles) else titles[i]
                a.set_title(cur_title)
            else:
                a.set_title(f"Image {i}")

    # removing axis
    if(not add_axis):
        for i in range(n_cols * n_rows):
            row, col = i // n_cols, i % n_cols
            a = ax[row, col] if n_rows > 1 else ax[col]
            a.axis("off")

    plt.tight_layout()
    if savepath is not None:
        plt.savefig(savepath)

    return fig, ax, ims


def add_border(x, color_name, pad=1):
    """
    Adding border to image frames

    Args:
    -----
    x: numpy array
        image to add the border to
    color_name: string
        Name of the color to use
    pad: integer
        number of pixels to pad each side
    """
    b, nc, h, w = x.shape
    zeros = torch.zeros if torch.is_tensor(x) else np.zeros
    px = zeros((b, 3, h+2*pad, w+2*pad))
    color = colors.to_rgb(color_name)
    px[:, 0, :, :] = color[0]
    px[:, 1, :, :] = color[1]
    px[:, 2, :, :] = color[2]
    if nc == 1:
        for c in range(3):
            px[:, c, pad:h+pad, pad:w+pad] = x[:, 0]
    else:
        px[:, :, pad:h+pad, pad:w+pad] = x
    return px


def overlay_segmentations(frames, segmentations, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on a sequence of images
    """
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, segmentation in zip(frames, segmentations):
        img = overlay_segmentation(frame, segmentation, colors, num_classes, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_segmentation(img, segmentation, colors, num_classes, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)
    seg_masks = (segmentation[0] == torch.arange(num_classes)[:, None, None].to(segmentation.device))
    img_with_seg = draw_segmentation_masks(
            img,
            masks=seg_masks,
            alpha=alpha,
            colors=colors
        )
    return img_with_seg / 255


def overlay_instances(frames, instances, colors, alpha):
    """
    Overlay instance segmentations on a sequence of images
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors
    if frames.max() <= 1:
        frames = frames * 255
    frames = frames.to(torch.uint8)

    imgs = []
    for frame, instance in zip(frames, instances):
        img = overlay_instance(frame, instance, colors, alpha)
        imgs.append(img)
    imgs = torch.stack(imgs)
    return imgs


def overlay_instance(img, instance, colors, alpha=0.7):
    """
    Overlaying the segmentation on an image
    """
    if colors[0] != "white":  # background should always be white
        colors = ["white"] + colors
    if img.max() <= 1:
        img = img * 255
    img = img.to(torch.uint8)
    instance_ids = instance.unique()
    instance_masks = (instance[0] == instance_ids[:, None, None].to(instance.device))
    cur_colors = [colors[idx.item()] for idx in instance_ids]
    img_with_seg = draw_segmentation_masks(
            img,
            masks=instance_masks,
            alpha=alpha,
            colors=cur_colors
        )
    return img_with_seg / 255


def qualitative_evaluation(imgs, targets, preds, unnorm=True):
    """
    Displaying the original images, target segmentation, and predicted segmentation
    """
    targets_vis = (targets * 255).long()
    targets_vis = overlay_segmentations(
            frames=imgs,
            segmentations=targets_vis,
            colors=VOC_COLORMAP,
            num_classes=23,
            alpha=1
        )
    preds_vis = overlay_segmentations(
            frames=imgs,
            segmentations=preds,
            colors=VOC_COLORMAP,
            num_classes=23,
            alpha=1
        )

    if unnorm:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        imgs = imgs * std + mean

    imgs, targets, preds = imgs[:6], targets[:6], preds[:6]
    fig, ax = plt.subplots(nrows=3, ncols=6)
    fig.set_size_inches(30, 10)
    ax[0, 0].set_ylabel("Images", fontsize=24)
    ax[1, 0].set_ylabel("Targets", fontsize=24)
    ax[2, 0].set_ylabel("Predictions", fontsize=24)
    for i in range(6):
        ax[0, i].imshow(imgs[i].detach().cpu().permute(1, 2, 0))
        ax[1, i].imshow(targets_vis[i].detach().cpu().permute(1, 2, 0))
        ax[2, i].imshow(preds_vis[i].detach().cpu().permute(1, 2, 0))
    for aa in ax:
        for a in aa:
            a.set_yticks([], [])
            a.set_xticks([], [])
    plt.tight_layout()
    return fig, ax

#