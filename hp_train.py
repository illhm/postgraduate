import avalanche
from avalanche import models
import torch
import torch.nn as nn
import os
import argparse

from avalanche.training import Naive
from h_prompt import Hierachical_Prompt_Plugin

import datasets
from clip import clip
from hp_clip import HP_CLIP
from datasets.caltech101_ava import Caltech101

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    if torch.cuda.is_available() and args.use_cuda:
        torch.backends.cudnn.benchmark = True
    # 1. 加载数据集
    caltech=Caltech101(args)
    train_stream,test_stream=caltech.getStreams()

    hp=HP_CLIP()
    # 2. 加载训练模型
    model = hp.get_model(args,caltech.get_class_names())
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)  # 使用的gpu id从执行命令中指定不在代码中显示指定
    model.to(device)
    optimizer = hp.get_optimizer(args,[])#todo []
    scheduler = hp.build_lr_scheduler(optimizer,args)
    # criterion是计算损失函数的，loss
    criterion = nn.CrossEntropyLoss()

    #算法指代的是strategy，strategy下面是plugin
    strategy_our = Naive(model=model,optimizer=optimizer, criterion=criterion,train_epochs=24,
                  train_mb_size=64,
                  plugins=[Hierachical_Prompt_Plugin(),
                           ],
                  device=device)

    result = []
    strategy = strategy_our
    for exp_id, experience in enumerate(train_stream):  # experience流
        print("taskId={},classes={}".format(experience.task_label, experience.classes_in_this_experience))
        strategy.train(experience, eval_streams=[])
        result.append(strategy.eval(test_stream[0:exp_id + 1]))
    torch.save(strategy.model.state_dict(), "trained_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="D:/dataset", help="parent path of dataset")
    parser.add_argument("--dataset", type=str, default="Caltech101", help="name of dataset")
    parser.add_argument("--gpt_dir", type=str, default="D:/project/2024-AAAI-HPT-main/data/gpt_data", help="path of gpt's data,including classes,descriptions,sturctures")
    parser.add_argument("--output-dir", type=str, default="", help="log output directory")
    parser.add_argument("--pool_size", type=int, default=20, help="log output directory")
    parser.add_argument("--description_num", type=int, default=5, help="log output directory")
    parser.add_argument("--t_prompt_length", type=int, default="", help="log output directory")
    parser.add_argument("--v_prompt_length", type=int, default="", help="log output directory")
    parser.add_argument("--output-dir", type=str, default="", help="log output directory")
    parser.add_argument("--use_cuda", type=bool, default=True)
    parser.add_argument(
        "--resume-model-path",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, help="only positive value enables a fixed seed"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="D:/project/2024-AAAI-HPT-main/configs/trainers/HPT/b2n.yaml", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="D:/project/2024-AAAI-HPT-main/configs/datasets/b2n/caltech101.yaml",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="HPT", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="ViT-B/16", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", default=True,action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="D:/project/2024-AAAI-HPT-main/results/output/B2N/train_base/caltech101/HPT/b2n_shots_16/seed1",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", default=10,type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    args = parser.parse_args()
    print("the run parameters:")
    main(args)