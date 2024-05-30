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
from gpt_generation import structure
from clip import clip



def parse_option():
    parser = argparse.ArgumentParser('Prompt Learning for CLIP', add_help=False)

    parser.add_argument("--root", type=str, default='/home/qc/dataset', help='parent path of dataset')
    parser.add_argument("--mean_per_class", action='store_true', help='mean_per_class')
    parser.add_argument("--num_runs", type=int, default=10, help='num_runs')
    parser.add_argument("--seed", type=int, default=0, help='random seed')
    parser.add_argument("--dataset", type=str, default="Caltech101", help="name of dataset")
    parser.add_argument("--gpt_dir", type=str, default="/home/qc/AttriCLIP-main/data/gpt_data", help="path of gpt's data,including classes,descriptions,sturctures")
    parser.add_argument("--pool_size", type=int, default=20, help="log output directory")
    parser.add_argument("--description_num", type=int, default=5, help="log output directory")
    parser.add_argument("--output-dir", type=str, default="/home/qc/AttriCLIP-main/result", help="log output directory")
    parser.add_argument("--use_cuda", type=bool, default=True)

    parser.add_argument("--backbone", type=str, default='ViT-L/14', help='path of backbone model file')
    parser.add_argument("--backbone_path", type=str, default='/home/qc/pretrained_model/ViT-L-14.pt', help='path of backbone model file')
    parser.add_argument("--keyprompt_path", type=str, default=None, help='path of keyprompt file')
    parser.add_argument("--save_path", type=str, default='/home/qc/AttriCLIP-main/result', help='path to save run results')
    parser.add_argument("--ID", type=str, default='description', help='description')

    # optimization setting
    parser.add_argument("--lr", type=float, default=0.01, help='num_runs')
    parser.add_argument("--wd", type=float, default=0.0, help='num_runs')
    parser.add_argument("--epochs", type=int, default=10, help='num_runs')
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
    parser.add_argument("--v_prompt_length", type=int, default=10, help='prompt length')
    parser.add_argument("--t_prompt_length", type=int, default=10, help='prompt length')

    parser.add_argument("--text_prompt", type=int, default=5, help='text_prompt')
    parser.add_argument("--keep", type=bool, default=False, help='keep')  # continue from other datasets

    args, unparsed = parser.parse_known_args()
    args.save_path = args.save_path + '/' + args.dataset + '/' + args.ID

    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main(args):
    # 1. 读取数据集，设置为增量形式
    inc_dataset = incremental_dataloader.IncrementalDataset(args=args, random_order=False,
                                                            # random class
                                                            shuffle=True, seed=args.seed, workers=8, validation_split=0, )

    # 2. 加载训练模型
    setup_seed(args.seed)
    algorithm = CoOp(args)
    allClass_tokenMap = getAllClassEmbeddings(algorithm.clip_model, args, inc_dataset.class_names)

    # 3. 开始训练过程,对于每一个增量任务
    for task in range(args.start_task, args.num_task):
        # 3.1 获取增量任务数据
        train_loader, test_loader, task_classnames, allSeen_classnames, task_info  = inc_dataset.get_task_data(task)

        # 3.3 打印训练参数
        print('task {} start to fit, taskInfo:{}'.format(task,str(task_info)))
        print("sample_per_task_testing",inc_dataset.sample_per_task_testing)  # dict{task:len(test)}
        # 3.2 处理任务所需额外信息，entities和cls token
        task_info['current_task'] =  task
        task_info['taskEntitys'] =  getTaskEntitys(args, task_classnames)
        task_info['allClass_tokenMap'] =  allClass_tokenMap

        # 3.4 增量task数据送入模型开始增量训练
        algorithm.train(train_loader, task_classnames, task_info)

        acc = algorithm.accuracy(test_loader, args.num_test, allSeen_classnames, mean_per_class=args.mean_per_class)
        print('test acc', acc)
        # save_result(args,algorithm,acc)


def getTaskEntitys(args, train_classnames):
    class_structures = structure.get_Classes_Structures(args, train_classnames)
    entities = []
    for classname, info in class_structures.items():
        for j in info:
            entities.extend(j["Entities"] + j["Attributes"])
    keys = set(i.lower() for i in entities)
    return keys

def getAllClassEmbeddings(clip_model, args, class_names,n_ctx=12):
    dtype = clip_model.dtype
    name_lens = [len(clip._tokenizer.encode(name)) for name in class_names]

    prompt_prefix = ' '.join(['x'] * n_ctx * args.text_prompt)
    prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]
    tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

    with torch.no_grad():
        embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(dtype)
        print("token_embedding output shape={}".format(embedding.shape))
    token_prefix= embedding[:, :1, :]
    token_suffix= embedding[:, 1 + (n_ctx * args.text_prompt):, :]

    # 与class无关的
    nc_prompts = [prompt_prefix + '.']
    nc_tokenized_prompts = torch.cat([clip.tokenize(p) for p in nc_prompts])
    with torch.no_grad():
        embedding = clip_model.token_embedding(nc_tokenized_prompts.cuda()).type(dtype)
    nc_token_prefix=embedding[:, :1, :]
    nc_token_suffix=embedding[:, 1 + n_ctx:, :]

    return { "name_lens":name_lens , "token_prefix":token_prefix,"token_suffix":token_suffix,
            "tokenized_prompts":tokenized_prompts}

def save_result(args,algorithm,acc):
    # 3.5 保存训练模型及（key，prompt）等运行结果
    if not os.path.isdir(args.save_path):
        mkdir_p(args.save_path)
    np.save(args.save_path + "/seed.txt", args.seed)
    torch.save(algorithm.model.state_dict()['entity'], os.path.join(args.save_path, 'text_key.pth.tar'))
    torch.save(algorithm.model.prompt_learner.state_dict()['text_prompt'],
               os.path.join(args.save_path, 'text_prompt.pth.tar'))

    # 3.6 运行数据指标保存
    with open(args.save_path + "/acc_task_" + str(args.current_task) + ".pickle", 'wb') as handle:
        pickle.dump(acc, handle, protocol=pickle.HIGHEST_PROTOCOL)



if __name__ == '__main__':
    if torch.cuda.is_available():
        print("gpunum={}".format(torch.cuda.device_count()))
    args = parse_option()
    main(args)
