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

__all__ = ["available_models", "load", "tokenize"]
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

def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):
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
        model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
    elif os.path.isfile(name):
        # 1.
        model_path = name 
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")
        
    try:
        # 2. loading JIT archive
        model = torch.jit.load(model_path, map_location=device if jit else "cpu").eval()
        state_dict = None
    except RuntimeError:
        # loading saved state dict
        if jit:
            warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
            jit = False
        state_dict = torch.load(model_path, map_location="cpu")

    if not jit:
        # 3.
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
        # 1.1 组装前缀，得到形如x x x x 的内容
        prompt_prefix = ' '.join(['x'] * n_ctx * self.args.text_prompt)
        # 1.2 得到形如['x x x x x x x x x x face.',...]的内容
        prompts = [prompt_prefix + ' ' + name + '.' for name in class_names]
        classnames = [name.replace('_', ' ') for name in class_names]
        # 1.3 得到classname对应的token词数
        self.name_lens = [len(_tokenizer.encode(name)) for name in class_names]
        self.prompt_pos = prompt_pos

        self.text_prompt = text_prompt
        # tokenization就是将原始文本切分成子单元，通常所说的分词，分出的每一个词语我们把它称为token
        tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
        self.tokenized_prompts = tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('token_prefix', embedding[:, :1, :])
        self.register_buffer('token_suffix', embedding[:, 1 + (n_ctx * self.args.text_prompt):, :])

        nc_prompts = [prompt_prefix + '.']
        nc_tokenized_prompts = torch.cat([tokenize(p) for p in nc_prompts])
        self.nc_tokenized_prompts = nc_tokenized_prompts
        with torch.no_grad():
            embedding = clip_model.token_embedding(nc_tokenized_prompts.cuda()).type(self.dtype)
        self.register_buffer('nc_token_prefix', embedding[:, :1, :])
        self.register_buffer('nc_token_suffix', embedding[:, 1 + n_ctx:, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.ctx_dim = ctx_dim

    def forward(self, indexs, test_class=False, infer=False):
        if test_class:
            prompt_prefix = ' '.join(['x'] * self.n_ctx * self.args.text_prompt)
            prompts = [prompt_prefix + ' ' + name + '.' for name in test_class]
            self.name_lens = [len(_tokenizer.encode(name)) for name in test_class]

            self.prompt_pos = self.prompt_pos

            tokenized_prompts = torch.cat([tokenize(p) for p in prompts])
            self.tokenized_prompts = tokenized_prompts
            with torch.no_grad():
                embedding = self.clip_model.token_embedding(tokenized_prompts.cuda()).type(self.dtype)
            self.register_buffer('token_prefix', embedding[:, :1, :])  # SOS, [n_cls, 1, ctx_dim]
            self.register_buffer('token_suffix', embedding[:, 1 + (self.n_ctx * self.args.text_prompt):,
                                                 :])  # CLS, EOS, [n_cls, -1, ctx_dim]
            self.n_cls = len(test_class)
        # 1. indexs是选中的prompt的索引集合
        batch = indexs.shape[0]
        # 1.1 取出对应的prompt
        ctx = self.text_prompt[indexs].view(batch, self.n_ctx * self.args.text_prompt, self.ctx_dim)
        tokenized_prompts = self.tokenized_prompts.view(self.n_cls, -1)
        n_cls = self.n_cls

        if self.prompt_pos == 2:
            # 增加维度，并在该维度复制tensor，repeat每个参数代表该维度重复的次数
            prefix = self.token_prefix.unsqueeze(0).repeat(batch, 1, 1, 1)
            suffix = self.token_suffix.unsqueeze(0).repeat(batch, 1, 1, 1)
            ctx = ctx.unsqueeze(1).repeat(1, n_cls, 1, 1)
            prompts = torch.cat([prefix, ctx, suffix], dim=2)
        elif self.prompt_pos == 1:
            prompts = []
            half_n_ctx = self.n_ctx // 2
            for i in range(n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i + 1, :, :].unsqueeze(1)
                class_i = self.token_suffix[i:i + 1, :name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i + 1, name_len:, :].unsqueeze(1)
                ctx_i_half1 = ctx[:, :half_n_ctx, :].unsqueeze(0)
                ctx_i_half2 = ctx[:, half_n_ctx:, :].unsqueeze(0)
                prompt = torch.cat([prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.prompt_pos == 0:
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = self.token_prefix[i:i + 1, :, :].unsqueeze(1)
                class_i = self.token_suffix[i:i + 1, :name_len, :].unsqueeze(1)
                suffix_i = self.token_suffix[i:i + 1, name_len:, :].unsqueeze(1)
                ctx_i = ctx.unsqueeze(0)
                prompt = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=2)
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        prompts = prompts.squeeze(2).view(batch * self.n_cls, -1, self.ctx_dim)
        tokenized_prompts = tokenized_prompts.unsqueeze(0).repeat(batch, 1, 1).view(batch * self.n_cls, -1)
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
        nc_prompts = torch.cat([prefix, ctx, suffix], dim=1)
        return nc_prompts, nc_tokenized_prompts

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    # text encoder对text输出的处理在这里
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

        """ 1. text encoder在这里实现 """
        self.text_encoder = TextEncoder(clip_model)
        if torch.cuda.device_count() > 1:
            self.text_encoder = nn.DataParallel(self.text_encoder)

        # 2. text_key 和text_prompt分开了，
        self.text_key = text_key
        # 2.1 只将text_prompt送入promptLearner进行学习
        self.prompt_learner = PromptLearner(self.args, class_names, clip_model, text_prompt, n_ctx=n_ctx)
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

        # 2. test过程
        if test:
            n_test = len(test_class)
            probability = image_features @ self.text_key.t()
            _, indexs = probability.topk(k=min(self.args.text_prompt, probability.shape[1]), dim=1, largest=True)

            # 通过 prompt_learner来学习prompt，再送入encoder
            text_prompt, tokenized_prompts = self.prompt_learner(indexs, test_class, test)
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            text_features = text_features.view(image_features.shape[0], n_test, -1)
            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)
            return logits
        # 3. train过程
        else:
            # 3.1 image和key计算相似度，取topK个 key; probability（32,10）             @运算，矩阵相乘，等价于np.matmul(A, B)；t运算：转置
            probability = image_features @ self.text_key.t()
            # 3.2 矩阵.topk()方法，找出矩阵中最大（小）的k个元素及它们的索引
            _, indexs = probability.topk(k=min(self.args.text_prompt, probability.shape[1]), dim=1, largest=True)
            # 3.3 取出匹配的k个key
            chosen_keys = self.text_key[indexs]
            # 3.4 这里执行prompt_learner的forward方法，把选中的prompt的index传进去，
            text_prompt, tokenized_prompts, nc_prompts, nc_tokenized_prompts = self.prompt_learner(indexs)

            # 3.5 将text送入 text encoder,得到text feature； text prompt：（320,77,768）；text features：（32,10,768） todo 有两次text encoder调用，nc代表什么？
            text_features = self.text_encoder(text_prompt, tokenized_prompts)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            text_features = text_features.view(image_features.shape[0], self.n_class, -1)

            image_features = image_features.unsqueeze(1)
            logit_scale = self.logit_scale.exp()
            logits = logit_scale * (image_features * text_features).sum(-1)

            #loss_m是正交损失，它是对text prompt经过text encoder得到的embedding来计算正交距离的；       nc_prompts（10,77,768），nc_text_features（10,768）
            prompt_embeddings = self.text_encoder(nc_prompts, nc_tokenized_prompts)
            prompt_embeddings = prompt_embeddings / prompt_embeddings.norm(dim=-1, keepdim=True)
            #dis（10,10）
            dis = prompt_embeddings @ prompt_embeddings.permute(1, 0)
            # torch.eye(bool)：对角为True，其余为False；  ~：对数据的每个二进制位取反
            loss_m = dis[~torch.eye(self.args.num_prompt, dtype=torch.bool, device='cuda')].abs().mean()

            return logits, image_features, chosen_keys, loss_m

    @property
    def dtype(self):
        return self.image_encoder.conv1.weight.dtype
