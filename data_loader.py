from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import torch
from PIL import Image
import numpy as np
import pandas as pd
from abc import abstractmethod
from PIL import ImageEnhance


transformtypedict=dict(Brightness=ImageEnhance.Brightness, Contrast=ImageEnhance.Contrast, Sharpness=ImageEnhance.Sharpness, Color=ImageEnhance.Color)


class ImageJitter(object):
    def __init__(self, transformdict):
        self.transforms = [(transformtypedict[k], transformdict[k]) for k in transformdict]


    def __call__(self, img):
        out = img
        randtensor = torch.rand(len(self.transforms))

        for i, (transformer, alpha) in enumerate(self.transforms):
            r = alpha*(randtensor[i]*2.0 -1.0) + 1
            out = transformer(out).enhance(r).convert('RGB')

        return out


def default_image_loader(path):
    return Image.open(path).convert('RGB')


class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]


class TransformLoader:
    def __init__(self, image_size,
                 normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)
        if transform_type == 'RandomSizedCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False, add_jitter=True, to_PIL=False):
        if to_PIL:
            transform_list = ['ToPILImage']
        else:
            transform_list = []

        if aug and add_jitter:
            transform_list += ['RandomSizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        elif aug and not add_jitter:
            transform_list += ['RandomSizedCrop', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list += ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


identity = lambda x:x


categories = ['Shoes', 'Boots', 'Sandals', 'Slippers']
closures = ['Lace up', 'Slip-On', 'Zipper', 'Hook and Loop', 'Pull-on']
genders = ['Women', 'Men', 'Girls', 'Boys']
brands = ['Frye', 'Nike', 'Nine West', 'SKECHERS', 'Merrell', 'Cole Haan', 'Stuart Weitzman', 'PUMA', 'New Balance',
          'UGG', 'Primigi Kids', 'Sperry Top-Sider', 'Naot Footwear', 'Sanuk', 'ASICS', 'Clarks', 'Anne Klein',
          'Stride Rite', 'Born', 'Keen']
shapes = ["triangle", "square", "plus", "circle", "tee"]
textures = ["solid", "stripes", "grid", "hexgrid", "dots"]
shapes_v = ["rhombus", "pentagon", "star", "fivesquare", "trapezoid"]
textures_v = ["noise", "triangles", "zigzags", "rain", "pluses"]
target_dict = {'Category':  categories, 'Closure': closures, 'Gender': genders, 'Brand': brands, 'shape_t': shapes,
               'texture_t': textures, 'shape_v': shapes_v, 'texture_v': textures_v}

class SimpleDataset:
    def __init__(self, data_file_meta, split, transform, targets=['Category', 'Closure', 'Gender'],
                 data_path='/data/ddmg/xray_data/zappos50k_data/ut-zap50k-images/', target_transform=identity):
        df = pd.read_csv(data_file_meta)
        self.meta = df[df['split'] == split]
        self.meta = self.meta.reset_index(drop=True)
        self.transform = transform
        self.target_transform = target_transform
        self.targets = targets
        self.data_path = data_path

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['filename'][i])
        img = Image.open(self.data_path + image_path).convert('RGB')
        img = self.transform(img)
        target = []
        for target_cat in self.targets:
            if target_cat in target_dict:
                target_list = target_dict[target_cat]
                if self.meta[target_cat].iloc[i] in target_list:
                    target.append(target_list.index(self.meta[target_cat].iloc[i]))
                else:
                    target.append(np.NaN)
            else:
                target.append(self.meta[target_cat].iloc[i])
        if len(target) == 1:
            target = target[0]
        return img, target

    def __len__(self):
        return self.meta.shape[0]


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, split, aug):
        pass


class SimpleDataManager(DataManager):
    def __init__(self, image_size, batch_size, targets=['Category', 'Closure', 'Gender'],
                 data_path='/data/ddmg/xray_data/zappos50k_data/ut-zap50k-images/', supcon=False):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.supcon = supcon
        self.targets = targets
        self.data_path = data_path

    def get_data_loader(self, data_file, split, aug, add_jitter=True):
        transform = self.trans_loader.get_composed_transform(aug, add_jitter)
        if self.supcon:
            transform = TwoCropTransform(transform)
        dataset = SimpleDataset(data_file, split, transform, self.targets, self.data_path)
        data_loader_params = dict(batch_size = self.batch_size, shuffle=True, num_workers = 12, pin_memory = True)
        data_loader = torch.utils.data.DataLoader(dataset, **data_loader_params)

        return data_loader
