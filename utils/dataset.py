import os
import numpy as np
import torch
from torchvision import transforms as T
import pydicom

from settings.config import Config
from utils.transformations import BoxedToTensor


class RSNA_Dataset(torch.utils.data.Dataset):
    """ Custome dataset for RSNA kaggle competition with DICOM data file format. """
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, idx):
        # base path to image
        img_path = os.path.join(Config.train_data_path, self.df.iloc[idx]['patientId'])
        img_path = img_path + '.dcm'
        # extract info about box and target
        target = self.df.iloc[idx, 1:].to_dict()
        # extract image array from DICOM file format
        image = pydicom.dcmread(img_path).pixel_array
        sample = (image, target)
        # perfome transformation if provided
        if self.transform:
            image, target = self.transform((image, target))
        else:
            image, target =  BoxedToTensor()((image, target))
        # add fine name to target dict
        target['file_name'] = self.df.iloc[idx, 0]
        # create sample of features and target    
        sample = (image, target)
        return sample
        