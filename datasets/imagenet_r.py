import os
from .imagenet import ImageNet
from data_utils import listdir_nohidden,Datum

TO_BE_IGNORED = ["README.txt"]


class ImageNetR():
    """ImageNet-R(endition).
    200 classes,30000 images
    """

    dataset_dir = "imagenet-r"

    def __init__(self, args):
        root = os.path.abspath(os.path.expanduser(args.dataset_root))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        # self.image_dir = os.path.join(self.dataset_dir, "imagenet-r")
        self.image_dir = self.dataset_dir

        text_file = os.path.join(self.dataset_dir, "classnames.txt")
        classnames = ImageNet.read_classnames(text_file)

        data = self.read_data(classnames)


    def read_data(self, classnames):
        image_dir = self.image_dir
        folders = listdir_nohidden(image_dir, sort=True)
        folders = [f for f in folders if f not in TO_BE_IGNORED]
        items = []

        for label, folder in enumerate(folders):
            imnames = listdir_nohidden(os.path.join(image_dir, folder))
            classname = classnames[folder]
            for imname in imnames:
                impath = os.path.join(image_dir, folder, imname)
                item = Datum(impath=impath, label=label, classname=classname)
                items.append(item)

        return items
