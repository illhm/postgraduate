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

    def __init__(self, args, random_order=False, shuffle=True, workers=8, seed=1,
                  validation_split=0.):
        """获取各个数据集的加载类及transform"""
        loadclass_of_dataset = _get_dataset_class(args.dataset)
        self.args = args
        """真正加载数据集图片，并实例化train和test 的dataset对象"""
        self._setup_data(
            loadclass_of_dataset,
            args.root,
            random_order=random_order,
            seed=seed,
            validation_split=validation_split
        )

        self._workers = workers
        self._shuffle = shuffle
        self.sample_per_task_testing = {}

    def _setup_data(self, loadclass_of_dataset, dataset_dir, random_order=False, seed=1,  validation_split=0.):
        """加载指定数据集，得到train、test、eval，"""
        self.incre_steps = []
        self.class_order = []
        # 1. 传入数据集路径，加载所有图片
        load_class = loadclass_of_dataset(dataset_dir)
        # 2. 获取三个subset及所有classname
        self.train_dataset, self.test_dataset, self.eval_dataset = load_class.getTrainTestEval_Dataset()
        self.class_names = load_class.get_class_names()

        # 3. 生成class的order列表——这里是按原顺序，可以自行更改
        order = [i for i in range(self.args.num_class)]
        if random_order:
            random.seed(seed)
            random.shuffle(order)
        self.class_order.append(order)
        # 4. 计算每个增量步骤中的class数量，list长度就是任务总数量，——self.increments里存的是每个task中的class数量，比如[10,10,10]
        self.incre_steps = [self.args.class_per_task for _ in range(len(order) // self.args.class_per_task)]

    @property
    def taskNum(self):
        return len(self.incre_steps)

    def get_classnames(self):
        return self.class_names

    def get_dataIndexs_by_labels(self, dataset, label_list, mode="train"):
        """从data_list中将符合指定label_list的数据全部提取出来"""
        indexs = []
        for i,item in enumerate(dataset.get_dataList()):
            if item[1] in label_list:
                indexs.append(i)

        return indexs


    def get_dataIndexs_of_test(self, dataset, label, mode="test"):
        """获取test数据，范围是训练至今的所有任务"""
        label_indexs = []
        label_targets = []

        np_indexs = np.array(list(range(len(dataset.get_dataList()))), dtype="uint32")
        np_target = np.array([i[1] for i in dataset.get_dataList()], dtype="uint32")

        #
        for t in range(len(label) // self.args.class_per_task):
            task_ids = []
            for class_id in label[t * self.args.class_per_task: (t + 1) * self.args.class_per_task]:
                idx = np.where(np_target == class_id)[0]
                task_ids.extend(list(idx.ravel()))
            task_ids = np.array(task_ids, dtype="uint32")
            task_ids.ravel()
            random.shuffle(task_ids)

            label_indexs.extend(list(np_indexs[task_ids]))
            label_targets.extend(list(np_target[task_ids]))
            if (t not in self.sample_per_task_testing.keys()):
                self.sample_per_task_testing[t] = len(task_ids)
        label_indexs = np.array(label_indexs, dtype="uint32")
        label_indexs.ravel()
        return list(label_indexs), label_targets

    def get_task_data(self, current_task):
        print("current_task：{},class_num：{}".format(current_task, self.incre_steps[current_task]))
        # 1. 获取本task中的class的序号范围
        min_class = sum(self.incre_steps[:current_task])
        max_class = sum(self.incre_steps[:current_task + 1])
        # 2. 根据序号范围获取class名称
        train_class_names = self.class_names[min_class:max_class]
        test_class_names = self.class_names[:max_class]
        # todo list(range(min_class, max_class))->[order[i] for in range(min_class, max_class)]
        train_indexs = self.get_dataIndexs_by_labels(self.train_dataset, list(range(min_class, max_class)),
                                                                 mode="train")
        self.train_data_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.args.train_batch,
                                                             shuffle=False, num_workers=self._workers,
                                                             #Sampler参数用于指定如何对数据进行采样，即确定每个批次中包含哪些样本。SubsetRandomSampler意为从给定的索引子集中进行随机采样
                                                             sampler=SubsetRandomSampler(train_indexs, False),drop_last=True)

        test_indexs, _ = self.get_dataIndexs_of_test(self.test_dataset, list(range(max_class)), mode="test")
        self.test_data_loader = torch.utils.data.DataLoader(self.test_dataset, batch_size=self.args.test_batch,
                                                            shuffle=False, num_workers=self._workers,
                                                            sampler=SubsetRandomSampler(test_indexs, False),drop_last=True)
        #没处理val_data_loader
        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "current_task": current_task,
            "max_task": len(self.incre_steps),
            "n_train_data": len(train_indexs),
            "n_test_data": len(test_indexs)
        }

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

