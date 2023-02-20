import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import torch
from torchvision import transforms as T

from settings.config import Config
from utils.transformations import BoxedToTensor, BoxedResize, BoxedRandomResizedCrop, BoxedCenterCrop, BoxedRandomHorizontalFlip, BoxedNormalize


def prepare_labels(df):
    """ Perfome some manipulation on data to combine boundarind boxes of same images

    Args:
        df (pd.DataFrame): raw dataset with on boundary box record per row
        
    Returns:
        df: preprocessed dataset
    """
    def create_boxes(row):
        return [row['x0'], row['y0'], row['x1'], row['y1']]
    # remove NaN's
    df = df.fillna(1)
    # convert [x, y, width, height] to [x0, y0, x1, y1]
    df['width'] = df['x'] + df['width']
    df['height'] = df['y'] + df['height']
    df = df.rename(columns={'x': 'x0', 'y': 'y0', 'width': 'x1', 'height': 'y1', 'Target': 'labels'})
    df['boxes'] = df.apply(create_boxes, axis=1)
    # group boxes of the same image to tuple
    df = df.groupby('patientId').agg({'boxes': tuple, 'labels': tuple})
    return df
    

  
def set_seed(seed):
    """ Set the same seed to all randomness.

    Args:
        seed (int): random seed
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    
    
def get_transfrom(subspace):
    """ Generate composed set of transformations for a given dataset split.

    Args:
        subspace (string): name of dataset subspace (train/ valid/ test)

    Returns:
        torchvision.transforms.transforms.Compose: set of transformation made on data
    """
    if subspace == 'train':
        transformations = T.Compose([
            BoxedToTensor(),
            BoxedResize(size=Config.resize_to),
            BoxedRandomResizedCrop(size=Config.img_size, scale=Config.random_scale),
            BoxedRandomHorizontalFlip(Config.flip_probability),
            BoxedNormalize(Config.mean, Config.std)
        ])
    else:
        transformations = T.Compose([
            BoxedToTensor(),
            BoxedResize(size=Config.resize_to),
            BoxedCenterCrop(size=Config.img_size),
            BoxedNormalize(Config.mean, Config.std)
        ])
    return transformations


def collate_fn(batch):
    return tuple(zip(*batch))


def imshow(img, target=None, title=None):
    """ Imshow for Tensor data type with localization box

    Args:
        img (torch.tensor): the image array in tensor data type
        target (dict): dict that contain info about target and box. Default to None.
        title (sting, optional): Title to the image. Defaults to None.
    """
    # change remove channel axis, convert to numpy 
    img = torch.squeeze(img).numpy()
    # restore image from normalization
    img = img * Config.std + Config.mean
    # remove axis
    plt.axis('off')
    # plot grayscale image
    plt.imshow(img, cmap=plt.cm.gray)
    # plot target box if exist
    if target:
        if (target['labels'] != 0).all():   # check if there 1 label
            for box in target['boxes']:
                plt.gca().add_patch(Rectangle((box[0], box[1]), box[2] - box[0], 
                                              box[3] - box[1], edgecolor='red', facecolor='none'))
    # show title if provided
    if title is not None:
        plt.title(title)
        
        
def show_grid(imgs, targets=None, columns=4, title=False, figsize=(15,15)):
    """ Plot grid of given image batch

    Args:
        imgs (torch.tensor): batch of images to plot
        targets (dict): dict with info about targets and boxes of each image. Defaults to None.
        columns (int): number of images per row. Defaults to 4.
        title (bool): if to plot title. Default to False. 
        figsize (tuple, optional): figure size of the image grid. Defaults to (15,15).
    """
    fig = plt.figure(figsize=figsize)
    for i in range(len(imgs)):
        rows = np.ceil(Config.batch_size / columns).astype(np.int16)
        fig.add_subplot(rows, columns, i + 1)
        title = targets[i]['file_name'] if title else None
        if targets:
            imshow(imgs[i], targets[i], title=title)
        else:
            imshow(imgs[i], title=title)
    plt.tight_layout()