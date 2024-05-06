import random
import random

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data.sampler import SubsetRandomSampler

from datasets.caltech101_ava import SplitCaltech
from .imagenet100 import imagenet100

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import collections


class SubsetRandomSampler(Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indexs (sequence): a sequence of indices
    """

    def __init__(self, indexs, shuffle):
        self.indexs = indexs
        self.shuffle = shuffle

    def __iter__(self):
        if (self.shuffle):
            return (self.indexs[i] for i in torch.randperm(len(self.indexs)))
        else:
            return (self.indexs[i] for i in range(len(self.indexs)))

    def __len__(self):
        return len(self.indexs)


class IncrementalDataset:

    def __init__(self, args, random_order=False, shuffle=True, workers=8, batch_size=128, seed=1,
                 increment=10, validation_split=0.):
        """获取各个数据集的加载类及transform"""
        dataset_class = _get_dataset_class(args.db_name)
        self.args = args
        """真正加载数据集图片，并实例化train和test 的dataset对象"""
        self._setup_data(
            dataset_class,
            args.root,
            random_order=random_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split
        )

        self._current_task = 0
        self._batch_size = batch_size
        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}

    def _setup_data(self, dataset_class, dataset_dir, random_order=False, seed=1, increment=10, validation_split=0.):
        self.increments = []
        self.class_order = []
        all_data_loaded = dataset_class(dataset_dir)
        self.train_dataset, self.test_dataset, self.eval_dataset = all_data_loaded.getTrainTest_Dataset()
        self.class_names = all_data_loaded.get_class_names()

        # 生成class的order列表
        order = [i for i in range(self.args.num_class)]
        if random_order:
            random.seed(seed)
            random.shuffle(order)
        self.class_order.append(order)
        # self.increments里存的是每个task中的class数量，比如[10,10,10]
        self.increments = [increment for _ in range(len(order) // increment)]

    @property
    def n_tasks(self):
        return len(self.increments)

    """从data_list中将符合指定label_list的数据全部提取出来"""
    def get_dataIndexs_by_labels(self, dataset, label_list, mode="train"):
        indexs = []
        for i,item in enumerate(dataset.get_dataList()):
            if item[1] in label_list:
                indexs.append(i)

        return indexs

    """获取test数据，范围是训练至今的所有任务"""
    def get_data_index_test(self, dataset, label, mode="test"):
        label_indices = []
        label_targets = []

        np_indexs = np.array(list(range(len(dataset.get_dataList()))), dtype="uint32")
        np_target = np.array([i[1] for i in dataset.get_dataList()], dtype="uint32")

        for t in range(len(label) // self.args.class_per_task):
            task_ids = []
            for class_id in label[t * self.args.class_per_task: (t + 1) * self.args.class_per_task]:
                idx = np.where(np_target == class_id)[0]
                task_ids.extend(list(idx.ravel()))
            task_ids = np.array(task_ids, dtype="uint32")
            task_ids.ravel()
            random.shuffle(task_ids)

            label_indices.extend(list(np_indexs[task_ids]))
            label_targets.extend(list(np_target[task_ids]))
            if (t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_ids)
        label_indices = np.array(label_indices, dtype="uint32")
        label_indices.ravel()
        return list(label_indices), label_targets

    def new_task(self):
        print("current_task：{},class_num：{}".format(self._current_task,self.increments[self._current_task]))
        # 1. 获取本task中的class的序号范围
        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])
        # 2. 根据序号范围获取class名称
        train_class_names = self.class_names[min_class:max_class]
        test_class_names = self.class_names[:max_class]
        # todo list(range(min_class, max_class))->[order[i] for in range(min_class, max_class)]
        train_indexs = self.get_dataIndexs_by_labels(self.train_dataset, list(range(min_class, max_class)),
                                                                 mode="train")
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self._batch_size,
                                                             shuffle=False, num_workers=8,
                                                             #Sampler参数用于指定如何对数据进行采样，即确定每个批次中包含哪些样本。SubsetRandomSampler意为从给定的索引子集中进行随机采样
                                                             sampler=SubsetRandomSampler(train_indexs, False))

        test_indexs, _ = self.get_data_index_test(self.test_dataset, list(range(max_class)), mode="test")
        self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,
                                                            shuffle=False, num_workers=8,
                                                            sampler=SubsetRandomSampler(test_indexs, False))
        #没处理val_data_loader
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "current_task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": len(train_indexs),
            "n_test_data": len(test_indexs)
        }

        self._current_task += 1

        return  self.train_data_loader, self.test_data_loader,train_class_names, test_class_names,  task_info

    def get_galary(self, task, batch_size=10):
        indexes = []
        dict_ind = {}
        seen_classes = []
        for i, t in enumerate(self.train_dataset.targets):
            if not (t in seen_classes) and (
                    t < (task + 1) * self.args.class_per_task and (t >= (task) * self.args.class_per_task)):
                seen_classes.append(t)
                dict_ind[t] = i

        od = collections.OrderedDict(sorted(dict_ind.items()))
        for k, v in od.items():
            indexes.append(v)

        data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False,
                                                  num_workers=4, sampler=SubsetRandomSampler(indexes, False))

        return data_loader

    def get_custom_loader_idx(self, indexes, mode="train", batch_size=10, shuffle=True):

        if (mode == "train"):
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=4, sampler=SubsetRandomSampler(indexes, True))
        else:
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=4, sampler=SubsetRandomSampler(indexes, False))
        return data_loader

    def get_custom_loader_class(self, class_id, mode="train", batch_size=10, shuffle=False):

        if (mode == "train"):
            train_indices, for_memory = self.get_dataIndexs_by_labels(self.train_dataset.targets, class_id, mode="train",
                                                                      memory=None)
            data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=4, sampler=SubsetRandomSampler(train_indices, True))
        else:
            test_indices, _ = self.get_dataIndexs_by_labels(self.test_dataset.targets, class_id, mode="test")
            data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False,
                                                      num_workers=4, sampler=SubsetRandomSampler(test_indices, False))

        return data_loader

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    def get_memory(self, memory, for_memory, seed=1):
        random.seed(seed)
        memory_per_task = self.args.memory // ((self.args.current_task + 1) * self.args.class_per_task)
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        mu = 1

        # update old memory
        if (memory is not None):
            data_memory, targets_memory = memory
            data_memory = np.array(data_memory, dtype="int32")
            targets_memory = np.array(targets_memory, dtype="int32")
            for class_idx in range(self.args.class_per_task * (self.args.current_task)):
                idx = np.where(targets_memory == class_idx)[0][:memory_per_task]
                self._data_memory = np.concatenate([self._data_memory, np.tile(data_memory[idx], (mu,))])
                self._targets_memory = np.concatenate([self._targets_memory, np.tile(targets_memory[idx], (mu,))])

        # add new classes to the memory
        new_indices, new_targets = for_memory

        new_indices = np.array(new_indices, dtype="int32")
        new_targets = np.array(new_targets, dtype="int32")
        for class_idx in range(self.args.class_per_task * (self.args.current_task),
                               self.args.class_per_task * (1 + self.args.current_task)):
            idx = np.where(new_targets == class_idx)[0][:memory_per_task]
            self._data_memory = np.concatenate([self._data_memory, np.tile(new_indices[idx], (mu,))])
            self._targets_memory = np.concatenate([self._targets_memory, np.tile(new_targets[idx], (mu,))])

        print(len(self._data_memory))
        return list(self._data_memory.astype("int32")), list(self._targets_memory.astype("int32"))


def _get_datasets(dataset_names):
    return [_get_dataset_class(dataset_name) for dataset_name in dataset_names.split("-")]


def _get_dataset_class(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "caltech101":
        return SplitCaltech
    if dataset_name == "imagenet100":
        return iIMAGENET100
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


class DataHandler:
    base_dataset = None
    train_transforms = []
    mata_transforms = [transforms.ToTensor()]
    common_transforms = [transforms.ToTensor()]
    class_order = None


class iIMAGENET100(DataHandler):
    base_dataset = imagenet100
    train_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]
    common_transforms = [
        transforms.Resize(224, interpolation=BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ]

