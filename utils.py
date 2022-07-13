import os
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt


class ImagePath(Dataset):
    def __init__(self, path, size=None):
        self.size = size

        self.images = [os.path.join(path,file) for file in os.listdir(path)]
        self._length = len(self.images)

        self.rescaler = albumentations.SmallestMaxSize(max_size=self.size)
        self.cropper = albumentations.CenterCrop(height=self.size, width=self.size)
        self.preprocessor = albumentations.Compose([self.rescaler,
                                                    self.cropper
                                                    ])

    def __len__(self):
        return self._length

    def preprocess_image(self,image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocess(image=image)["image"]
        image = (image/127.5 - 1.0).astype(np.float32)
        image = image.transpose(2,0,1)
        return image

    def __getitem__(self, i):
        example = self.preprocess_image(self.images[i])
        return example

def load_data(args):
    train_data = ImagePath(args.dataset_path,size=256)
    train_loader = DataLoader(train_data,batch_size=args.batch_size,shuffle=True)
    return train_loader


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') !=-1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') !=-1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def plot_image(image:dict):
    x = image["imputs"]
    reconstruction = image["reconstruction"]
    sample_half = image["sample_half"]
    sample_nopix = image["sample_nopix"]
    sample_det = image["sample_det"]

    fig, axarr = plt.subplots(1,5)
    axarr[0].imshow()
    axarr[1].imshow(reconstruction)
    axarr[2].imshow(sample_half)
    axarr[3].imshow(sample_nopix)
    axarr[4].imshow(sample_det)
    plt.show()