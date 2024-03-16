import os
from torchvision import transforms
from .data_utils import read_split, read_and_split_data, save_split, get_lab2cname
from torch.utils.data import dataset
from torch.utils.data.dataset import Dataset, T_co
from PIL import Image

IGNORED = ["BACKGROUND_Google", "Faces_easy"]
NEW_CNAMES = {
    "airplanes": "airplane",
    "Faces": "face",
    "Leopards": "leopard",
    "Motorbikes": "motorbike",
}


class MyDataset(Dataset):

    def __init__(self, dataList):
        self.dataList = dataList
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            lambda image: image.convert("RGB"),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        self._lab2cname, self._classnames = get_lab2cname(dataList)

    def __getitem__(self, i):  # return the ith data in the set.
        item = self.dataList[i]
        # 在这里加载图片文件
        image_path=item[0]
        image = Image.open(image_path).convert('RGB')
        image_trans = self.transform(image)
        label=item[1]
        classname=item[2]
        return image_trans, label,classname


    def __len__(self):
        return len(self.dataList)

    def len(self):
        return len(self.dataList)

    def get_dataList(self):
        return self.dataList

    def get_classnames(self):
        return self._classnames

class SplitCaltech():
    images_folder = "caltech-101"
    def __init__(self,dataset_dir):
        self._images = None
        root = os.path.abspath(os.path.expanduser(dataset_dir))
        self.dataset_dir = os.path.join(root, self.images_folder)
        self.folders_dir = os.path.join(self.dataset_dir, "101_ObjectCategories")
        self.split_path = os.path.join(self.dataset_dir, "split_zhou_Caltech101.json")

        if os.path.exists(self.split_path):
            train, val, test = read_split(self.split_path, self.folders_dir)
        else:
            train, val, test = read_and_split_data(self.folders_dir, ignored=IGNORED,new_cnames=NEW_CNAMES)
            save_split(train, val, test, self.split_path, self.folders_dir)

        self.train_set, self.val, self.test_set = MyDataset(train), MyDataset(val), MyDataset(test)
        self._lab2cname, self._classnames = get_lab2cname(train)

    def getTrainTest_Dataset(self):
        return self.train_set, self.test_set, self.val

    def get_class_names(self):
        return self._classnames
