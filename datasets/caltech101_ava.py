import os
import pickle
import json
import os.path as osp
import errno
import math
import random
from .data_utils import read_split,read_and_split_data,read_json,save_split,get_lab2cname,Datum
from avalanche.benchmarks import nc_benchmark

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}

from torch.utils.data.dataset import Dataset, T_co
from avalanche.benchmarks.utils import PathsDataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
#加载Caltech101数据集
class Caltech101(Dataset):
    images_folder = "caltech-101"

    def __init__(self, dataset_dir,train_transform,test_transform):
        # root = os.path.abspath(os.path.expanduser(args.DATASET.ROOT))
        # self.dataset_dir = os.path.join(root, args.dataset_dir)
        self._images = None
        root = os.path.abspath(os.path.expanduser(dataset_dir))
        self.dataset_dir = os.path.join(root, self.images_folder)
        self.image_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        if os.path.exists(self.split_path):
            train, val, test = read_split(self.split_path, self.image_dir,train_transform)
        else:
            train, val, test = read_and_split_data(self.image_dir, train_transform, ignored=IGNORED, new_cnames=NEW_CNAMES)
            save_split(train, val, test, self.split_path, self.image_dir)#参照上面，加入transform，改动返回内容，

        self.train_set, self.val, self.test_set=train, val, test
        self._lab2cname, self._classnames = get_lab2cname(self.train_set)

    def getStreams(self):
        return self.train_set,self.test_set,self.val

    def get_class_names(self):
        return self._classnames

    def __getitem__(self, index):
        pass   # todo