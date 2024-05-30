import os
import hashlib
import urllib
import warnings
from typing import Any, Union, List
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import torch
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

from .model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

if torch.__version__.split(".") < ["1","7","1"]:
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

__all__ = ["available_models", "load_and_build_clip"]
_tokenizer = _Tokenizer()

_MODELS = {
    "RN50": "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    "RN101": "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    "RN50x4": "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
    "RN50x16": "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
    "RN50x64": "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
    "ViT-B/32": "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    "ViT-B/16": "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    "ViT-L/14": "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    }

def _download(url:str,root:str):
    os.makedirs(root,exist_ok=True)
    filename = os.path.basename(url)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root,filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")
        
    if os.path.isfile(download_target):
        if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if hashlib.sha256(open(download_target,"rb").read()).hexdigest()!= expected_sha256:
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target

def _transform(n_px):
    return Compose([
        Resize(n_px,interpolation=BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073),(0.26862954, 0.26130258, 0.27577711)),
    ])

def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())

def load_and_build_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name:str
        A model name listed by `clip.available_models()", or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit: bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model: torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    if name in _MODELS:
        # 1. 下载
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # 2. 预先下载好放到服务器上了
        model_path = name 
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
        
    try:
        # 3. 加载loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        # 4. 根据加载得到的模型的state_dict（包括参数和持久缓冲区，以(key,value)形式给出）来组装clip模型
        model = build_model(state_dict or model.state_dict()).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda:torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim: :Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs =[]

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device)=="cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()


        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]: # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())

class PromptLearner(nn.Module):
    def __init__(self, args, class_names, clip_model, entity_prompts, n_ctx=12, prompt_pos=2):
        super().__init__()
        ctx_dim = clip_model.ln_final.weight.shape[0]
        dtype = clip_model.dtype
        self.clip_model = clip_model
        self.args = args
        self.dtype = dtype
        self.prompt_pos = prompt_pos
        self.entity_prompts = entity_prompts
        self.task_cls_num = len(class_names)
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim


    def forward(self, indexs , task_info , test_class = None, test = False):

        if test:
            # prompt_prefix =' '.join(['x'] * self.n_ctx*self.args.text_prompt)
            # prompts = [prompt_prefix + ' ' + name + '.' for name in test_class]
            # self.name_lens = [len(_tokenizer.encode(name)) for name in test_class]
            # self.prompt_pos = self.prompt_pos
            # tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
            # self.tokenized_prompts = tokenized_prompts
            # with torch.no_grad():
            #     embedding = self.clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
            # self.register_buffer( 'token_prefix', embedding[:, :1, :]) # SOS, [n_cls, 1, ctx_dim]
            # self.register_buffer( 'token_suffix', embedding[:, 1+(self.n_ctx*self.args.text_prompt):,:]) # CLS, EOS, [n_cls, -1, ctx_dim]
            self.task_cls_num = len(test_class)

        current_task = task_info['current_task']
        allClass_tokenMap = task_info['allClass_tokenMap']
        self.name_lens = allClass_tokenMap['name_lens']
        self.token_prefix = allClass_tokenMap['token_prefix']
        self.token_suffix = allClass_tokenMap['token_suffix']
        self.tokenized_prompts = allClass_tokenMap['tokenized_prompts'][ : (current_task+1) * self.args.class_per_task]

        # 1. indexs是选中的prompt的索引集合
        batch = indexs.shape[0]
        # 1.1 取出对应的prompt,ctx:[32,60,768],name_embeddings:[10,768]
        ctx = self.entity_prompts[indexs].view(batch, self.n_ctx * self.args.text_prompt, self.ctx_dim)

        prompts = []
        # 取出当前见过的所有task的cls token，与当前task所选择的prompt进行一一拼装
        for i in range( (current_task+1) * self.args.class_per_task ):
            name_len = self.name_lens[i]
            # Expand prefix, class, and suffix to match the batch size
            prefix_i = self.token_prefix[i:i + 1, :, :].expand(batch, -1, -1)
            class_i = self.token_suffix[i:i + 1, :name_len, :].expand(batch, -1, -1)
            suffix_i = self.token_suffix[i:i + 1, name_len:, :].expand(batch, -1, -1)

            # Expand ctx_i to match batch size and class length
            ctx_i = ctx.unsqueeze(1)

            # Concatenate along the third dimension
            prefix_i = prefix_i.unsqueeze(1)
            class_i = class_i.unsqueeze(1)
            suffix_i = suffix_i.unsqueeze(1)
            # print("prefix_i shape:", prefix_i.shape)
            # print("class_i shape:", class_i.shape)
            # print("ctx_i shape:", ctx_i.shape)
            # print("suffix_i shape:", suffix_i.shape)
            # Concatenate along the second dimension for each class
            prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
            prompts.append(prompt)

            # Concatenate all prompts and adjust dimensions
        prompts = torch.cat(prompts, dim=0)
        #todo 下面的cls_num是当前taks的还是所有的
        prompts = prompts.view(batch * self.task_cls_num, -1, self.ctx_dim)

        tokenized_prompts = self.tokenized_prompts.view(self.task_cls_num, -1)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).expand(batch, -1, -1).contiguous().view(batch * self.task_cls_num, -1)

        self.prompts = prompts
        self.prompts_token = tokenized_prompts

        # if test:
        #     return prompts, tokenized_prompts
        # else:
        #     nc_prompts, nc_tokenized_prompts = self.only_prefix()
        #     return prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts

        return prompts, tokenized_prompts

    def only_prefix(self):
        ctx = self.entity_prompts
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
        self.token_embedding = clip_model.token_embedding
        self.tokenzie = clip_model.to
        self.dtype = clip_model.dtype

    # text encoder对text输出的处理在这里
    def forward(self, x, tokenized_prompts):#text encoder,prompt.shape=torch.Size([107, 77, 768]),tokenized_prompt.shape=torch.Size([107, 77]),positional_embedding.shape=torch.Size([77, 768])
        print("text encoder,prompt.shape={},tokenized_prompt.shape={},positional_embedding.shape={}".format(x.shape,tokenized_prompts.shape,self.positional_embedding.shape))
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def encode_text(self, texts):
        x = self.token_embedding(texts).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), texts.argmax(dim=-1)] @ self.text_projection
        return x
        # outs=[]
        # for text in texts:
        #     x= self.token_embedding(text).type(self.dtype) # [batch_size, n_ctx, d_model]
        #
        #     x= x+ self.positional_embedding.type(self.dtype)
        #     x= x.permute(1,0,2)# NLD -> LND
        #     x= self.transformer(x)
        #     x= x.permute(1, 0, 2) # LND -> NLD
        #     x= self.ln_final(x).type(self.dtype)
        #
        #     # x.shape = [batch_size, n_ctx, transformer.width]
        #     # take features from the eot embedding (eot_token is the highest number in each sequence)
        #     x = x[torch.arange(x.shape[0]),text.argmax(dim=-1)] @ self.text_projection
        #     outs.append(x)
        #
        # return outs

class TextEncoderZS(nn.Module):
    """ path 1：frozen text encoder"""
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.token_embedding = clip_model.token_embedding

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)

        feats = []
        for _, layer in enumerate(self.transformer):
            x = layer(x)
            # save class embeddings from different layers
            feats.append(x[text.argmax(dim=-1), torch.arange(x.shape[1])])

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        txt_feats = torch.stack(feats)

        return x, txt_feats


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ---------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate:bool
        whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """

    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class CLIP(nn.Module):
    def __init__(self, args, task_classnames, clip_model, task_info, n_ctx=12):
        super().__init__()
        self.n_class = len(task_classnames)
        self.args = args
        self.task_info = task_info
        entities = task_info["taskEntitys"]

        """ 1. text encoder在这里实现 """
        self.text_encoder = TextEncoder(clip_model)
        # self.text_encoder_zs = TextEncoderZS(args,clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)

        # 2.将attribute送入text encoder，得到feautre，作为prompt key；
        with torch.no_grad():
            # 遇到DataParallel’ object has no attribute ‘xxxx’时，在model后面加上.module.
            # tokenized_keys = torch.cat([tokenize(p) for p in entities]).cuda()
            # entity_embeddings1 = clip_model.token_embedding(tokenized_keys)
            # entity_embeddings = [t/t.norm(dim=-1, keepdim=True) for t in entity_embeddings]
            tokenized_keys = torch.cat([tokenize(p).cuda() for p in entities])
            entity_embeddings = self.text_encoder.module.encode_text(tokenized_keys)
            entity_embeddings /= entity_embeddings.norm(dim=-1, keepdim=True)
            # 使用entity、attribute的text embedding作为prompt key，这里将entity注册为model的参数
            # entity_embeddings = nn.Parameter(entity_embeddings)
            print("encode_text entity_embeddings shape={}".format(entity_embeddings.shape))
            self.entity_keys = entity_embeddings

            # tokenize_names = [tokenize(c).cuda() for c in class_names]
            tokenize_names = torch.cat([tokenize(p).cuda() for p in task_classnames])
            name_embeddings = self.text_encoder.module.encode_text(tokenize_names)
            print("encode_text name_embeddings shape={}".format(name_embeddings.shape))
            name_embeddings /= name_embeddings.norm(dim=-1, keepdim=True)
            # name_embeddings = [t/t.norm(dim=-1, keepdim=True) for t in name_embeddings]
            self.name_embeddings = name_embeddings

        # 初始化对应数量的attribute prompt   ,n_ctx：上下文长度，ctx_dim：上下文维度 n_cls：类别数量 name_lens：每个类别名称的长度
        ctx_dim = clip_model.ln_final.weight.shape[0]
        entity_prompt = torch.empty(len(entities), n_ctx, ctx_dim, dtype=clip_model.dtype).cuda()
        nn.init.normal_(entity_prompt, std=0.02)
        entity_prompt = nn.Parameter(entity_prompt)
        self.entity_prompts = entity_prompt

        #
        self.prompt_learner = PromptLearner(self.args, task_classnames, clip_model, self.entity_prompts, n_ctx=n_ctx)
        # 3. image encoder，直接使用clip中的，未做修改
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale

    def forward(self, image, num_test=None, test_class=None, test=False):
        # 1. 图片送入image encoder，得到image feature
        with torch.no_grad():
            image_features = self.image_encoder(image.type(self.dtype))
            # 1.1 向量单位化
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.detach()

        if test:
            n_test = len(test_class)
            probability = image_features @ self.entity_keys.t()
            _, indices = probability.topk(k=min(self.args.text_prompt, probability.shape[1]), dim=1, largest=True)

            text_prompt, tokenized_prompts = self.prompt_learner(indices, self.task_info)
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            text_features = text_features.view(image_features.shape[0], n_test, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)
            return logits
        else:
            # 3.1 image和key计算相似度，取topK个 key; probability（32,260）batch,entityNum             @运算，矩阵相乘，等价于np.matmul(A, B)；t运算：转置
            probability = image_features @ self.entity_keys.t()
            # 3.2 矩阵.topk()方法，找出矩阵中最大（小）的k个元素及它们的索引
            _, indexs = probability.topk(k=min(self.args.text_prompt, probability.shape[1]), dim=1, largest=True)
            # 3.3 取出匹配的k个key  indexs:[32,5],chosen_keys:[32,5,768]
            chosen_keys = self.entity_keys[indexs]
            # 取出key对应的prompt，
            # 3.4 把选择的prompt与cls token起来，作为text encoder的输入
            # prompts, tokenized_prompts, nc_prompts, nc_tokenized_prompts= self.prompt_learner(indexs,self.task_info)
            prompts, tokenized_prompts = self.prompt_learner(indexs,self.task_info)
            #entity_prompts:[260,12,768] entity_keys:[260,768];final_text_input:(32,60,768) ;positional_embedding:[77,768]
            # 3.5 将text vector送入 text encoder,得到最终的text feature，用于cosine sim计算；
            text_features = self.text_encoder(prompts,tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(image_features.shape[0], self.n_class, -1)

            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)

            # todo 下面的loss要换成：   计算class相似度，据此引导所选择的key和prompt互相靠近相似
            #loss_m是正交损失，它是对text prompt经过text encoder得到的embedding来计算正交距离的；       nc_prompts（10,77,768），nc_text_features（10,768）

            return logits, image_features, chosen_keys

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
