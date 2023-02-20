import numpy as np
import torch
from torchvision.transforms import functional as TF

    
class BoxedToTensor(): 
    """ Custom transform to Tensor data type """
    def __call__(self, sample):
        """ Transform image and boundary box to Tensor data type

        Args:
            sample (tuple): image, dict with boundary box info

        Returns:
            img, target: tuple, transformed sample 
        """
        img, target = sample
        target = {k:torch.tensor(v, dtype=torch.float) for k, v in target.items()}
        img = torch.from_numpy(img).to(torch.float)
        img = torch.unsqueeze(img, 0)
        return img, target
    
    
class BoxedResize():
    """ Custom transform to resize image with boundary box """
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):
        """ Resize image to the given shape and also adapt boundary box

        Args:
            sample (tuple): image, dict with boundary box info

        Returns:
            img, target: tuple, transformed sample 
        """
        img, target = sample
        h, w = img.shape[-2:]
        # rescale image
        img = TF.resize(img, self.size)
        # find scale value of each axis
        scale_h, scale_w = img.shape[-2] / h, img.shape[-1] / w
        # rescale boundary box
        if (target['labels'] != 0).all():    # check if boxes exists
            target['boxes'][:, [0, 2]] *= scale_w
            target['boxes'][:, [1, 3]] *= scale_h
        return img, target
    
    
class BoxedRandomResizedCrop():
    """ Custom transform to randomly crop image with random scale and resizing to given shape with boundary box """
    def __init__(self, size, scale=(0.8, 1.0)):
        self.size = size
        self.scale = scale
        
    def __call__(self, sample):
        """ Create crop boundary of random scale, randomly crop image and resize to given shape with boundary box support

        Args:
            sample (tuple): image, dict with boundary box info

        Returns:
            img, target: tuple, transformed sample 
        """
        img, target = sample
        h, w = img.shape[-2:]
        # generate scale value from uniform distibution
        scale = np.random.uniform(self.scale[0], self.scale[1], 1)[0]
        # calculate shape of croped area
        new_h, new_w = (h * scale).astype(np.int16), (w * scale).astype(np.int16)
        # generate random padding for cropped area
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        # crop image
        img = img[:, top: top + new_h, left: left + new_w]
        # move boundary box
        if (target['labels'] != 0).all():    # check if boxes exists
            target['boxes'][:, [0, 2]] -= left
            target['boxes'][:, [1, 3]] -= top
        # create sample
        sample = (img, target)
        # resize sample
        img, target = BoxedResize(self.size)(sample)
        return img, target
        
   
class BoxedCenterCrop():
    """ Custom transform for center croping of image with boudary box support """
    def __init__(self, size):
        self.size = size
        
    def __call__(self, sample):
        """ Perfome center cropping of given image with respect to boundary box

        Args:
            sample (tuple): image, dict with boundary box info

        Returns:
            img, target: tuple, transformed sample 
        """
        img, target = sample
        h, w = img.shape[-2:]
        # perfome center cropping
        img = TF.center_crop(img, self.size)
        # move boundary box
        if (target['labels'] != 0).all():    # check if boxes exists
            target['boxes'][:, [0, 2]] -= (w - self.size[1]) / 2
            target['boxes'][:, [1, 3]] -= (h - self.size[0]) / 2
        # create sample
        sample = (img, target)
        return img, target 
    
    
class BoxedRandomHorizontalFlip():
    """ Custom transform for horizontal flipping """
    def __init__(self, prob=0.5):
        self.prob = prob
        
    def __call__(self, sample):
        """ Randomly flip img and boundaey box

        Args:
            sample (tuple): image, dict with boundary box info

        Returns:
            img, target: tuple, transformed sample 
        """
        img, target = sample
        # take probability of flipping
        flip = torch.bernoulli(torch.tensor(self.prob, dtype=torch.float))
        if flip:
            img = TF.hflip(img)
            # move box over x axis
            if (target['labels'] != 0).all():    # check if boxes exists
                target['boxes'][:, [0, 2]] = img.shape[-1] - target['boxes'][:, [0, 2]]
                target['boxes'][:, [0, 2]] = target['boxes'][:, [2, 0]]
        return img, target
    
    
class BoxedNormalize():
    """ Custom transform for image normalization """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        """ Perfome image normalization

        Args:
            sample (tuple): image, dict with boundary box info

        Returns:
            img, target: tuple, transformed sample 
        """
        img, target = sample
        img = TF.normalize(img, self.mean, self.std)
        target = {k:torch.tensor(v, dtype=torch.int64) for k, v in target.items()}
        return img, target
