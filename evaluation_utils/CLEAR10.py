import torch
import random
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable
from PIL import Image
from glob import glob


class CLEAR10IMG(Dataset):
    """ Learning CLEAR10 """

    def __init__(self, root_dir, bucket, form="all", split_ratio=0.7, debug=False, transform=None):
        '''
        Args: 
            root_dir(str list): folder path of 11 images
            bucket(int): time bucket id
            form(str): all -> whole dataset; train -> train dataset; test -> test dataset
            split_ratio(float, optional): proportion of train images in dataset
            transform(optional): transformation
        '''
        self.root_dir = root_dir
        self.transform = transform
        self.bucket = bucket
        self.form = form
        self.input_folders = self.root_dir+"/"+str(bucket+1)
        self.img_paths = list(filter(lambda x: x.endswith(".jpg"), 
                                     glob(self.input_folders + '/**',recursive=True)))
        
        # code classes by alphabetical order
        self.targets = [self.img_paths[idx][len(self.input_folders):].split("/")[1] 
                        for idx in range(len(self.img_paths))]
        classes_name = sorted(list(set(self.targets)))
        classes_code = range(len(classes_name))
        self.classes_mapping = dict(zip(classes_name,classes_code))
        self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
        
        if debug == True:
            self.img_paths = self.img_paths[:25]
            self.targets = self.targets[:25]
        if form != "all":
            self.train_img_paths = set(random.sample(self.img_paths,int(len(self.img_paths)*split_ratio)))
            self.test_img_paths = list(set(self.img_paths) - self.train_img_paths) 
            self.train_img_paths = list(self.train_img_paths)
            if form == "train":
                self.targets = [self.train_img_paths[idx][len(self.input_folders):].split("/")[1] 
                                for idx in range(len(self.train_img_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()
            else:
                self.targets = [self.test_img_paths[idx][len(self.input_folders):].split("/")[1] 
                                for idx in range(len(self.test_img_paths))]
                self.targets = torch.Tensor([self.classes_mapping[x] for x in self.targets]).int()

    def __len__(self): 
        if self.form == "all":
            return len(self.img_paths)
        elif self.form == "train":
            return len(self.train_img_paths)
        else:
            return len(self.test_img_paths)

    def __getitem__(self,idx):
        if self.form == "all":
            img = Image.open(self.img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.img_paths[idx][len(self.input_folders):].split("/")[1] # exclude the first empty entry
        elif self.form == "train":
            img = Image.open(self.train_img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.train_img_paths[idx][len(self.input_folders):].split("/")[1]
        else:
            img = Image.open(self.test_img_paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            label = self.test_img_paths[idx][len(self.input_folders):].split("/")[1]
        sample = {'img': img, 'target': self.classes_mapping[label]}
        if self.transform is not None:
            sample['img'] = self.transform(sample['img'])
        return sample['img'], sample['target']

