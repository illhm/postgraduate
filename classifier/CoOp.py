from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from clip.clip import load, tokenize, CLIP
from .utils import build_cosine_scheduler, cosine_loss

class CoOp:
    def __init__(self, prev_key, prev_prompt, args, n_ctx=12, use_float32=False, use_grad_checkpoint=False, keep=False):
        # 1. 从指定路径加载clip模型
        clip_model, _ = load(args.backbone_path)  # 1.1 这一句是Attriclip中的，单纯加载clip模型，没有做特殊结构改动
        # clip_model= hp_clip.load_clip(args=args,device=device)     # 1.2 这一句是HPT中的，
        # 1.1 将模型设置为Evaluation模式
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_prompt = args.num_prompt
        self.n_ctx = n_ctx
        self.lr = args.lr * args.train_batch / 20
        self.wd = args.wd
        self.epochs = args.epochs
        self.train_batch = args.train_batch
        self.args = args
        dtype = clip_model.dtype
        self.dtype = dtype

        # 2. Attribute Word Bank初始化
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # key 和 prompt 向量从正态分布中取值来进行初始化
        text_key = torch.empty(self.num_prompt, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(text_key, std=0.02)  # 从正态分布中取值，来初始化神经网络层的权重
        text_prompt = torch.empty(self.num_prompt, n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(text_prompt, std=0.02)
        # 2.1  key和prompt只是可学习的网络参数
        if keep == True:
            self.text_key = nn.Parameter(prev_key)
            self.text_prompt = nn.Parameter(prev_prompt)
        else:
            self.text_key = nn.Parameter(text_key)
            self.text_prompt = nn.Parameter(text_prompt)

    def count_acc(self, logits, label):
        # torch.argmax()函数：求最大数值的索引
        pred = torch.argmax(logits, dim=1)
        if torch.cuda.is_available():
            return (pred == label).sum().item()
        else:
            return (pred == label).sum().item()

    def fit(self, data_loader, class_names, task_info):
        # 这里代表每个epoch中有多少batch
        per_epoch_steps = len(data_loader)
        """ 1. 在这里将text key和prompt放入clip模型中"""
        self.init_clip_model(class_names=class_names, per_epoch_steps=per_epoch_steps, text_key=self.text_key,
                             text_prompt=self.text_prompt)
        self.model.eval()
        # 2. train过程，计算CE损失，匹配损失和正交损失
        for epoch in range(self.epochs):
            loop = tqdm(data_loader, total=len(data_loader))
            total_loss, total_match, total_num = 0.0, 0, 0
            for idx, (image_tensor, label, class_name) in enumerate(loop):
                label = label - self.args.class_per_task * task_info['current_task']
                # lab_idx = label.cpu().numpy().tolist()
                cur_iter_idx = epoch * per_epoch_steps + idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)
                # clip模型的forward函数在这里调用，具体函数见159行
                output, image_features, chosen_keys, loss_m = self.model(image_tensor.cuda())
                match = self.count_acc(output, label.cuda())
                # 根据结果，计算loss
                loss_main = F.cross_entropy(output, label.cuda())
                # loss_k用于将选中的keys和image embedding的距离拉近
                loss_k = cosine_loss(image_features, chosen_keys)
                loss = loss_main + 0.5 * loss_k + 0.1 * loss_m
                total_loss += loss
                # 梯度后向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_num += label.shape[0]
                total_match += match
                epoch_acc = float(match) / float(label.shape[0])
                print("match={},batch={}".format(match, label.shape[0]))
                loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
                loop.set_postfix(loss=loss.item(), acc=epoch_acc)  # 为进度条显示额外信息
            print("epoch:{}, train_accuracy={}, average_loss={}".format(epoch, float(total_match) / float(total_num),
                                                                        float(total_loss) / float(idx + 1)))

    def init_clip_model(self, class_names, per_epoch_steps, text_key, text_prompt):

        self.n_class = len(class_names)
        # 通过深拷贝clip模型，实现clip主体参数的冻结，只训练key和prompt
        clip_model = deepcopy(self.clip_model)
        """自己实现的clip模型，以方便修改结构，放入text key和prompt参数"""
        self.model = CLIP(self.args, class_names, clip_model, text_key, text_prompt, self.n_ctx)
        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True

        """在这里将text key和prompt的参数放入optimizer中进行优化"""
        Other_params = [param for name, param in self.model.named_parameters() if 'text_key' in name]
        param_dict = [{'params': [p for p in self.model.prompt_learner.parameters() if p.requires_grad]},
                      {'params': Other_params}]
        self.optimizer = torch.optim.SGD(param_dict, lr=self.lr, weight_decay=self.wd)
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=self.lr,
            total_step=self.epochs * per_epoch_steps)

    @torch.no_grad()
    def accuracy(self, loader, num_test, test_class, mean_per_class=False):
        if mean_per_class:
            return self._accuracy_mpc(loader, num_test, test_class)
        else:
            return self._accuracy(loader, num_test, test_class)

    def _accuracy_mpc(self, loader, num_test, test_class):
        n_class = self.n_class
        acc_per_class = [0 for _ in range(n_class)]
        count_per_class = [0 for _ in range(n_class)]
        for i, (image, label, class_name) in enumerate(loader):
            pred_y = self.inference(image.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            for c in range(n_class):
                acc_per_class[c] += ((top_labels.view(-1) == label.cuda()) * (label.cuda() == c)).sum().item()
                count_per_class[c] += (label.cuda() == c).sum().item()
        acc = [a * 1.0 / c for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()
        return acc

    def _accuracy(self, loader, num_test, test_class):
        total_count = 0
        acc_count = 0
        for i, (image, label, class_name) in enumerate(loader):
            pred_label = self.inference(image.cuda(), num_test, test_class)
            _, top_labels = pred_label.topk(1, dim=-1)
            acc_count += (top_labels.view(-1) == label.cuda()).sum().cpu().numpy()
            total_count += label.shape[0]
        acc = acc_count * 1.0 / total_count
        acc = acc.item()
        return acc

    @torch.no_grad()
    def inference(self, image, num_test, test_class):
        logits = self.model(image, num_test, test_class, test=True)
        return logits.float().softmax(dim=-1)
