from PIL import Image
import torch.utils.data as data_utils
import os
import re
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from collections import defaultdict
from utils.img_reader import read_image
import matplotlib.pyplot as plt
import torchvision.transforms as T

class BaseDataSet(Dataset):
    def __init__(self, img_source, transforms=None, mode="RGB"):
        self.mode = mode
        self.transforms = transforms
        self.root = os.path.dirname(img_source)
        assert os.path.exists(img_source), f"{img_source} NOT found."
        self.img_source = img_source

        self.label_list = list()
        self.path_list = list()
        self._load_data()
        self.label_index_dict = self._build_label_index_dict()

    def __len__(self):
        return len(self.label_list)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"| Dataset Info |datasize: {self.__len__()}|num_labels: {len(set(self.label_list))}|"

    def _load_data(self):
        with open(self.img_source, 'r') as f:
            for line in f:
                _path, _label = re.split(r",| ", line.strip())
                self.path_list.append(_path)
                self.label_list.append(_label)

    def _build_label_index_dict(self):
        index_dict = defaultdict(list)
        for i, label in enumerate(self.label_list):
            index_dict[label].append(i)
        return index_dict

    def __getitem__(self, index):
        path = self.path_list[index]
        img_path = os.path.join(self.root, path)
        label = self.label_list[index]

        img = read_image(img_path, mode=self.mode)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


