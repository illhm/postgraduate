import torch
import torch.nn as nn
from torch.nn import functional as F

from tqdm import tqdm
from copy import deepcopy
import numpy as np

from clip.clip_2 import load, tokenize
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
import dataset.incremental_dataloader

from .utils import build_cosine_scheduler, cosine_loss
import time

class PromptLearner(nn.Module):
    def __init__(self, args, class_names, clip_model, text_prompt, n_ctx=12, prompt_pos=2):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.clip_model = clip_model
        self.args = args
        n_cls = len(class_names)
        self.dtype = dtype
        # 1. 拼装prompt
        prompt_prefix =' '.join(['x'] * n_ctx * self.args.text_prompt)
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]
        classnames = [name.replace('_', ' ') for name in class_names]

        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        self.prompt_pos = prompt_pos

        self.text_prompt = text_prompt
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer( 'token_prefix', embedding[:, :1, :])
        self.register_buffer( 'token_suffix', embedding[:, 1+(n_ctx*self.args.text_prompt):,:])

        nc_prompts = [prompt_prefix+'.' ]
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1,:])
        self.register_buffer('nc_token_suffix', embedding[:, 1+n_ctx:,:])

        self.n_cls = n_cls 
        self.n_ctx = n_ctx 
        self.ctx_dim = ctx_dim

    def forward(self,indices, test_class=False, infer=False):
        if test_class:
            prompt_prefix =' '.join(['x'] * self.n_ctx*self.args.text_prompt)
            prompts = [prompt_prefix + ' ' + name + '.' for name in test_class]
            self.name_lens = [len(_tokenizer.encode(name)) for name in test_class]

            self.prompt_pos = self.prompt_pos

            tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
            self.tokenized_prompts = tokenized_prompts
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
            self.register_buffer( 'token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
            self.register_buffer( 'token_suffix', embedding[:, 1+(self.n_ctx*self.args.text_prompt):,:]) # CLS, EOS, [n_cls, -1, ctx_dim]
            self.n_cls = len(test_class)
        batch = indices.shape[0]
        ctx=self.text_prompt[indices].view(batch, self.n_ctx*self.args.text_prompt, self.ctx_dim)
        tokenized_prompts = self.tokenized_prompts.view(self.n_cls,-1)
        n_cls = self.n_cls

        if self.prompt_pos == 2:
            prefix = self.token_prefix.unsqueeze(0).repeat(batch,1,1,1)
            suffix = self.token_suffix.unsqueeze(0).repeat(batch,1,1,1)
            ctx = ctx.unsqueeze(1).repeat(1, n_cls, 1, 1)
            prompts = torch.cat([prefix, ctx, suffix],dim=2)
        elif self.prompt_pos == 1:
            prompts =[]
            half_n_ctx = self.n_ctx // 2
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1, :,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1,:name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i_half1 = ctx[:,:half_n_ctx, :].unsqueeze(0)
                ctx_i_half2 = ctx[:, half_n_ctx:,:].unsqueeze(0)
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.prompt_pos == 0:
            prompts =[]
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i+1,:,:].unsqueeze(1)
                class_i = self.token_suffix[i:i+1, :name_len,:].unsqueeze(1)
                suffix_i = self.token_suffix[i:i+1, name_len:,:].unsqueeze(1)
                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        prompts = prompts.squeeze(2).view(batch*self.n_cls, -1, self.ctx_dim)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch,1,1).view(batch*self.n_cls, -1)
        self.prompts = prompts
        self.prompts_token = tokenized_prompts
        if infer:
            return prompts, tokenized_prompts
        else:
            nc_prompts, nc_tokenized_prompts = self.only_prefix()
            return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

    def only_prefix(self):
        ctx = self.text_prompt
        prompt_size = ctx.shape[0]
        nc_tokenized_prompts = self.nc_tokenized_prompts.repeat(prompt_size, 1)
        prefix = self.nc_token_prefix.repeat(prompt_size, 1, 1)
        suffix = self.nc_token_suffix.repeat(prompt_size, 1, 1)
        nc_prompts = torch.cat([prefix, ctx, suffix],dim=1)
        return nc_prompts, nc_tokenized_prompts


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    #text encoder对text输出的处理在这里
    def forward(self, x, tokenized_prompts):
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x


class CLIP(nn.Module):
    def __init__(self, args, class_names, clip_model, text_key, text_prompt, n_ctx=12):
        super().__init__()
        self.n_class = len(class_names)
        self.args = args

        # 1. text enoder
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)

        self.prompt_learner = PromptLearner(self.args, class_names, clip_model, text_prompt, n_ctx=n_ctx)
        self.text_key = text_key
        # 3. image encoder
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, num_test=None, test_class=None, test=False):

        with torch.no_grad():
            # 1. 图片送入image encoder，得到image feature
            image_features = self.image_encoder(image.type(self.dtype))
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        # 2.1 test过程
        if test:
            n_test = len(test_class)
            probability = image_features @ self.text_key.t()
            _, indices = probability.topk(k=min(self.args.text_prompt,probability.shape[1]), dim=1, largest=True)

            # 通过 prompt_learner来学习prompt，再送入encoder
            text_prompt, tokenized_prompts = self.prompt_learner(indices,test_class,test)
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            text_features = text_features.view(image_features.shape[0], n_test, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)
            return logits
        # 2.2 train过程
        else:
            n_class = self.n_class
            # 2.2.1 image feature 和 attribute key 之间计算相似度，取topK个 key
            # 2.2.2 @运算，矩阵相乘，等价于np.matmul(A, B)
            probability = image_features @ self.text_key.t()
            # 2.2.3 矩阵.topk()方法，找出矩阵中最大（小）的k个元素及它们的索引，这里相当于是找出相似度最大的k个prompt key
            _, indices = probability.topk(k=min(self.args.text_prompt, probability.shape[1]), dim=1, largest=True)
            chosen_keys = self.text_key[indices]
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = self.prompt_learner(indices)
            # 将text送入 text encoder, todo 有两次text encoder调用，nc代表什么？
            text_features = self.text_encoder(text_prompt,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(image_features.shape[0], n_class, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)
           
            nc_text_features = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            nc_text_features = nc_text_features / nc_text_features.norm(dim=-1, keepdim=True)
            dis = nc_text_features @ nc_text_features.permute(1, 0)
            loss_m = dis[~torch.eye(self.args.num_prompt, dtype=torch.bool, device='cuda')].abs().mean()

            return logits, image_features, chosen_keys, loss_m


    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype

import hp_clip
class CoOp:
    def __init__(self, prev_key, prev_prompt,args, n_ctx=12, use_float32=False, use_grad_checkpoint=False, keep=False):
        # 1. 从指定路径加载clip模型
        # clip_model, _ = load(args.backbone_path)          # 1.1 这一句是Attriclip中的，单纯加载clip模型，没有做特殊结构改动
        clip_model= hp_clip.load_clip_to_cpu(args=args)     # 1.2 这一句是HPT中的，
        # 1.1 将模型设置为Evaluation模式
        clip_model.eval()
        if use_float32:
            clip_model.float()
        self.clip_model = clip_model
        self.use_grad_checkpoint = use_grad_checkpoint
        self.num_prompt = args.num_prompt
        self.n_ctx = n_ctx
        self.lr = args.lr*args.train_batch/20
        self.wd = args.wd
        self.epochs = args.epochs
        self.train_batch = args.train_batch 
        self.args = args
        dtype = clip_model.dtype
        self.dtype = dtype

        # prompt learner
        ctx_dim = clip_model.ln_final.weight.shape[0]
        # 2. （key，prompt）向量采用正态初始化
        text_key = torch.empty(self.num_prompt, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(text_key, std=0.02)#从正态分布中取值，来初始化神经网络层的权重
        text_prompt = torch.empty(self.num_prompt, n_ctx, ctx_dim, dtype=self.dtype).cuda()
        nn.init.normal_(text_prompt, std=0.02)
        # 2.1 将（key，prompt）转化为可学习的网络参数
        if  keep == True :
            self.text_key = nn.Parameter(prev_key)
            self.text_prompt = nn.Parameter(prev_prompt)
        else:
            self.text_key = nn.Parameter(text_key)
            self.text_prompt = nn.Parameter(text_prompt)

    def fit(self, data_loader,class_names,len_data):
        if len(data_loader.dataset)< self.train_batch:
            real_img_bsz = len(data_loader.dataset)
            self.lr = self.lr * real_img_bsz / self.train_batch 
        else:
            real_img_bsz = self.train_batch

        per_epoch_steps = len(data_loader)
        """ 1. 在这里将text key和prompt放入clip模型中"""
        self.init_self_model(class_names=class_names, per_epoch_steps=per_epoch_steps, text_key=self.text_key, text_prompt=self.text_prompt)
        self.model.eval()

        # 2. 正式训练，单张图片送入模型，根据输出，计算loss
        for epoch in range(self.epochs):
            for idx, (image_tensor,label, class_name) in enumerate(data_loader):
                
                label = label - self.args.class_per_task * self.args.current_task
                lab_idx = label.cpu().numpy().tolist()
                cur_iter_idx = epoch*per_epoch_steps+idx
                self.cur_iter_idx = cur_iter_idx
                self.scheduler.step(cur_iter_idx)
                #clip模型的forward函数在这里调用，具体函数见159行
                output, ima_feat, key_choose, loss_m = self.model(image_tensor.cuda())
                #根据结果，计算loss
                loss_main = F.cross_entropy(output, label.cuda())
                loss_k = cosine_loss(ima_feat,key_choose)
                loss = loss_main + 0.5*loss_k + 0.1*loss_m
                #梯度后向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()



    def init_self_model(self, class_names, per_epoch_steps, text_key, text_prompt):

        self.n_class = len(class_names)
        #把原生模型进行深拷贝，得到副本
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
            total_step=self.epochs*per_epoch_steps)

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
        for i, (image,label,class_name) in enumerate(loader):
            pred_y = self.inference(image.cuda())
            _, top_labels = pred_y.topk(1, dim=-1)
            for c in range(n_class):
                acc_per_class[c] += ((top_labels.view(-1) == label.cuda()) * (label.cuda()== c)).sum().item()
                count_per_class[c] += (label.cuda() == c).sum().item()
        acc = [a*1.0/c for (a, c) in zip(acc_per_class, count_per_class)]
        acc = np.array(acc).mean()
        return acc

    def _accuracy(self, loader, num_test, test_class):
        total_count=0
        acc_count =0
        for i,(image,label,class_name) in enumerate(loader):
            pred_label = self.inference(image.cuda(), num_test, test_class)
            _, top_labels = pred_label.topk(1, dim=-1)
            acc_count += (top_labels.view(-1)==label.cuda()).sum().cpu().numpy()
            total_count += label.shape[0]
        acc = acc_count*1.0/total_count
        acc = acc.item()
        return acc

    @torch.no_grad()
    def inference(self,image, num_test, test_class):
        logits = self.model(image, num_test, test_class, test=True)
        return logits.float().softmax(dim=-1)
