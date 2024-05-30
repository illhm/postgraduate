from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from clip.clip import load_and_build_clip, CLIP
from .utils import build_cosine_scheduler, cosine_loss
from clip import hp_clip
from clip.hp_clip import CustomCLIP

class CoOp:
    def __init__(self,  args, n_ctx=12, use_float32=False, use_grad_checkpoint=False):
        # 1. attriclip的load，从指定路径加载clip模型，并进行特定修改
        clip_model, _ = load_and_build_clip(args.backbone_path)
        # clip_model= hp_clip.load_clip_to_cpu(args=args).to(torch.device("cuda"))     # 1.2 这一句是HPT中的，
        # 1.1 将模型设置为Evaluation模式，和model.train()相对应，这对于某些特定类型的层是非常重要的，比如 BatchNorm 和 Dropout 层，它们在训练和评估阶段的行为是不同的
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


    def count_acc(self, logits, label):
        # torch.argmax()函数：求最大数值的索引
        pred = torch.argmax(logits, dim=1)
        if torch.cuda.is_available():
            return (pred == label).sum().item()
        else:
            return (pred == label).sum().item()

    def train(self, data_loader, task_classnames, task_info):
        # 这里代表每个epoch中有多少batch
        batch_num = len(data_loader)
        """ 1. 在这里将text key和prompt放入clip模型中"""
        # 对于每个task，将其中class对应的attribute、entity作为text_key，训练相应的text_prompt，再进行匹配，计算loss
        # todo 要适应cl，这里的text_key就需要是带继承的，不能每次只是该task生成的
        self.init_clip_model(task_classnames, batch_num, task_info)
        self.model.eval()
        # 2. train过程，计算CE损失，匹配损失和正交损失
        for epoch in range(self.epochs):
            loop = tqdm(data_loader, total=len(data_loader))
            loop.set_description(f'Epoch [{epoch}/{self.epochs}]')
            print(" ")
            total_loss, total_match, total_num = 0.0, 0, 0
            for idx, (images, labels, names) in enumerate(loop):
                labels = labels - self.args.class_per_task * task_info['current_task']
                # lab_idx = label.cpu().numpy().tolist()
                cur_iter_idx = epoch * batch_num + idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)
                # clip模型的forward函数在这里调用，具体函数见159行
                output, image_features, chosen_keys = self.model(images.cuda())
                match = self.count_acc(output, labels.cuda())
                # 根据结果，计算loss
                loss_main = F.cross_entropy(output, labels.cuda())
                # loss_k用于将选中的keys和image embedding的距离拉近
                # loss_k = cosine_loss(image_features, chosen_keys)
                # loss = loss_main + 0.5 * loss_k
                loss = loss_main
                total_loss += loss
                # 梯度后向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_num += labels.shape[0]
                total_match += match
                epoch_acc = float(match) / float(labels.shape[0])
                # print("match={},batch={}".format(match, labels.shape[0]))
                # loop.set_postfix(loss=loss.item(), acc=epoch_acc)  # 为进度条显示额外信息
            print("epoch:{}, train_accuracy={}, average_loss={}".format(epoch, float(total_match) / float(total_num),
                                                                        float(total_loss) / float(idx + 1)))

    def init_clip_model(self, task_classnames, batch_num, task_info):

        self.n_class = len(task_classnames)
        # 通过深拷贝clip模型，实现clip主体参数的冻结，只训练key和prompt
        clip_model = deepcopy(self.clip_model)

        # attriclip的，自己实现的clip模型，以方便修改结构，放入text key和prompt参数
        # todo 每次都copy一个新的model，有必要吗，不是只要冻结其梯度，只开启需训练参数的梯度即可吗
        self.model = CLIP(self.args, task_classnames, clip_model, task_info, self.n_ctx)
        # hpt的，
        # self.model = CustomCLIP(self.args,class_names,clip_model)

        if self.use_grad_checkpoint:
            try:
                self.model.text_encoder.transformer.use_gradient_checkpoint = True
            except:
                self.model.text_encoder.module.transformer.use_gradient_checkpoint = True

        """在这里将text key和prompt的参数放入optimizer中进行优化"""
        Other_params = [param for name, param in self.model.named_parameters() if 'entity_embeddings' in name]
        param_dict = [{'params': [p for p in self.model.prompt_learner.parameters() if p.requires_grad]},
                      {'params': Other_params}]
        self.optimizer = torch.optim.SGD(param_dict, lr=self.lr, weight_decay=self.wd)
        # self.optimizer = torch.optim.Adam(param_dict, lr=self.lr, weight_decay=self.wd)
        self.scheduler = build_cosine_scheduler(
            self.optimizer,
            lr=self.lr,
            total_step=self.epochs * batch_num)



    @torch.no_grad()
    def accuracy(self, loader, num_test, test_class, mean_per_class=False):
        if mean_per_class:
            return self._accuracy_mpc(loader, num_test, test_class)
        else:
            return self._accuracy(loader, num_test, test_class)

    #mpc 代表mean_per_class
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
