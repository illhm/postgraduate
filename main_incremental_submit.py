import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
import argparse
import pickle
import random
import numpy as np
import torch

import dataset.incremental_dataloader as incremental_dataloader
from classifier.CoOp import CoOp
from utils import mkdir_p


def parse_option():
    parser = argparse.ArgumentParser('Prompt Learning for CLIP', add_help=False)

    parser.add_argument("--root", type=str, default='/home/qc/dataset', help='parent path of dataset')
    parser.add_argument("--db_name", type=str, default='Caltech101', help='dataset name')
    parser.add_argument("--mean_per_class", action='store_true', help='mean_per_class')
    parser.add_argument("--num_runs", type=int, default=10, help='num_runs')
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--dataset", type=str, default="Caltech101", help="name of dataset")
    parser.add_argument("--gpt_dir", type=str, default="/home/qc/AttriCLIP-main/data/gpt_data", help="path of gpt's data,including classes,descriptions,sturctures")
    parser.add_argument("--pool_size", type=int, default=20, help="log output directory")
    parser.add_argument("--description_num", type=int, default=5, help="log output directory")
    parser.add_argument("--output-dir", type=str, default="/home/qc/AttriCLIP-main/result", help="log output directory")
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--backbone_path", type=str, default='/home/qc/pretrained_model/ViT-L-14.pt', help='path of backbone model file')
    parser.add_argument("--backbone", type=str, default='ViT-L/14', help='path of backbone model file')
    parser.add_argument("--keyprompt_path", type=str, default=None, help='path of keyprompt file')
    parser.add_argument("--save_path", type=str, default='/home/qc/AttriCLIP-main/result', help='path to save run results')
    parser.add_argument("--ID", type=str, default='description', help='description')

    # optimization setting
    parser.add_argument("--lr", type=float, default=1e-3, help='num_runs')
    parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
    parser.add_argument("--epochs", type=int, default=5, help='num_runs')
    parser.add_argument("--train_batch", type=int, default=32, help='num_runs')
    parser.add_argument("--test_batch", type=int, default=32, help='num_runs')

    # model setting
    parser.add_argument("--model", type=str, default='coop', help='model')
    parser.add_argument("--n_prompt", type=int, default=32, help='num_runs')
    parser.add_argument("--prompt_bsz", type=int, default=4, help='num_runs')

    # incremental setting
    parser.add_argument("--num_class", type=int, default=100, help='total num of classes')
    parser.add_argument("--class_per_task", type=int, default=10, help='class num per task')#增量步长
    parser.add_argument("--num_task", type=int, default=10, help='num_task')
    parser.add_argument("--start_task", type=int, default=0, help='start session')
    parser.add_argument("--current_task", type=int, default=0, help='current session')
    parser.add_argument("--memory", type=int, default=0, help='memory')
    parser.add_argument("--num_test", type=int, default=15, help='num_test_text')
    parser.add_argument("--num_prompt", type=int, default=10, help='num_prompt')
    parser.add_argument("--text_prompt", type=int, default=3, help='text_prompt')
    parser.add_argument("--keep", type=bool, default=False, help='keep')  # continue from other datasets

    args, unparsed = parser.parse_known_args()
    args.save_path = args.save_path + '/' + args.db_name + '/' + args.ID

    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    # 1. 读取数据集，设置为增量形式
    inc_dataset = incremental_dataloader.IncrementalDataset( args=args, random_order=False,
                                                            # random class
                                                            shuffle=True, seed=args.seed, batch_size=args.train_batch,
                                                            workers=8, validation_split=0,
                                                            #increment，增量步长
                                                            increment=args.class_per_task, )
    # 2. 加载训练模型
    # 2.1 是否使用预训练的（key，prompt）
    prev_key, prev_prompt = False, False
    setup_seed(args.seed)
    if args.keep == True:
        # 之前训练的结果
        path_key = os.path.join(args.keyprompt_path, 'text_key.pth.tar')
        path_prompt = os.path.join(args.keyprompt_path, 'text_prompt.pth.tar')
        prev_key = torch.load(path_key)
        prev_prompt = torch.load(path_prompt)
        print('prompt trained from previous dataset')
    else:
        print('prompt trained from random')
    # 2.2 在CoOp模型基础上进行定制
    if args.model == 'coop':
        model = CoOp(prev_key, prev_prompt, args=args, keep=args.keep)

    start_task = args.start_task
    memory = None
    # 3. 开始训练过程
    for task in range(start_task, args.num_task):
        # 3.1 获取增量任务及增量数据
        train_loader, test_loader, train_classnames, test_classnames, task_info  = inc_dataset.new_task()
        # 3.2 根据配置项，可实现接续训练
        if start_task != 0 and start_task == task:
            inc_dataset._current_task = task
            with open(args.save_path + "/sample_per_task_testing_" + str(task - 1) + ".pickle", 'rb') as handle:
                sample_per_task_testing = pickle.load(handle)
            inc_dataset.sample_per_task_testing = sample_per_task_testing

        # 打印训练参数
        print('task {} start to fit, taskInfo:{}'.format(task,str(task_info)))
        print("sample_per_task_testing",inc_dataset.sample_per_task_testing)  # dict{task:len(test)}

        # 3.3 增量task数据送入模型开始增量训练
        model.fit(train_loader, train_classnames, task_info)

        # 3.4 保存训练模型及（key，prompt）等运行结果
        if not os.path.isdir(args.save_path):
            mkdir_p(args.save_path)
        np.save(args.save_path + "/seed.txt", args.seed)
        torch.save(model.model.state_dict()['text_key'], os.path.join(args.save_path, 'text_key.pth.tar'))
        torch.save(model.model.prompt_learner.state_dict()['text_prompt'],
                   os.path.join(args.save_path, 'text_prompt.pth.tar'))
        acc = model.accuracy(test_loader, args.num_test, test_classnames, mean_per_class=args.mean_per_class)
        print('acc', acc)
        # 3.4 运行数据指标保存
        with open(args.save_path + "/memory_" + str(args.current_task) + ".pickle", 'wb') as handle:
            pickle.dump(memory, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(args.save_path + "/acc_task_" + str(args.current_task) + ".pickle", 'wb') as handle:
            pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(args.save_path + "/sample_per_task_testing_" + str(args.current_task) + ".pickle", 'wb') as handle:
            pickle.dump(inc_dataset.sample_per_task_testing, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    args = parse_option()
    main(args)
