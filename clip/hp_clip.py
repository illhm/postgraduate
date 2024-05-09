import clip.clip as clip
import torch
from torch import nn
import os.path as osp
import json
import random
import os
from gpt_generation import description,structure


"""加载 指定name的 原生clip模型,load_clip_to_cpu方法是hpt的"""
def load_clip_to_cpu(args):
    backbone_name = args.backbone
    url = clip._MODELS[backbone_name]
    # 1.有模型文件就用，没有就下载
    if os.path.isfile(args.backbone_path):
        model_path = args.backbone_path
    else:
        model_path = clip._download(url,root=args.backbone_path)
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    # 2.
    model = clip.build_model(state_dict or model.state_dict())
    return model

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

class VisionPromptLearner(nn.Module):
    """path 1：prompt learner for image encoder"""
    def __init__(self, args, clip_model):
        super().__init__()
        self.n_vpro = args.v_prompt_length
        self.pro_dim = clip_model.visual.ln_pre.weight.shape[0]
        self.dtype = clip_model.dtype
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding
        self.layers = len(clip_model.visual.transformer.resblocks)
        # global prompt for image encoder (except for the first layer)
        self.p_visual = nn.ParameterList([nn.Parameter(torch.empty(self.n_vpro, self.pro_dim).type(self.dtype))
                                          for _ in range(self.layers - 1)]).cuda()
        for p in self.p_visual:
            nn.init.normal_(p, std=0.02)

        # global prompt for the first layer of image encoder
        self.p_input = nn.Parameter(torch.empty(self.n_vpro, self.pro_dim)).cuda()
        nn.init.normal_(self.p_input, std=0.02)

    def forward(self, x):
        x = x.type(self.dtype)
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1],
                                                                      dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)

        # insert global visual prompt of the first layer
        p_input = self.p_input.unsqueeze(0).expand(len(x), -1, -1)
        x = torch.cat([x, p_input], dim=1)

        return x, self.p_visual

class VisionEncoder(nn.Module):
    """path 1：prompted image encoder"""
    def __init__(self, args, clip_model):
        super().__init__()
        visual = clip_model.visual
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.n_vpro = args.v_prompt_length  # prompt length

    def forward(self, x, p_visual):
        x = self.ln_pre(x).type(self.dtype)
        x = x.permute(1, 0, 2)

        for layer_idx, layer in enumerate(self.transformer):
            if layer_idx > 0:
                # insert layer-wise global visual prompt
                x[-self.n_vpro:] = p_visual[layer_idx - 1].unsqueeze(1).expand(-1, x.shape[1], -1)
            x = layer(x)

        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj

        return x

class VisionEncoderZS(nn.Module):
    """ path 2：frozen image encoder """
    def __init__(self, cfg, clip_model):
        super().__init__()

        visual = clip_model.visual
        self.ln_pre = visual.ln_pre
        self.transformer = visual.transformer.resblocks
        self.ln_post = visual.ln_post
        self.proj = visual.proj
        self.dtype = clip_model.dtype
        self.conv1 = clip_model.visual.conv1
        self.class_embedding = clip_model.visual.class_embedding
        self.positional_embedding = clip_model.visual.positional_embedding

    def forward(self, x):
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x).type(self.dtype)
        x = x.permute(1, 0, 2)

        x = self.transformer(x)

        x = x.permute(1, 0, 2)
        x = self.ln_post(x[:, 0, :])
        x = x @ self.proj

        return x

class TextEncoder(nn.Module):
    """ path 2：Hierachical Prompted Text Encoder"""
    def __init__(self, args, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer.resblocks
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_tpro = args.t_prompt_length  # prompt length
        self.n_set = args.description_num  # number of descriptions for each category

    def forward(self, x, high_prompt, global_prompt, tokenized_prompts, attn, flag):
        # prompt_high——p_ins: instance-specific prompt, a.k.a high-level prompt from descriptions
        # prompt_global——p_uni: task-unified prompt, a.k.a global-level prompt
        # flag: True when training and False when testing
        # Since we use all (self.n_set) descriptions for learning high-level prompt, we should reshape p_ins first.
        (l, c, d) = high_prompt.shape
        high_prompt = high_prompt.reshape(l, c // self.n_set, self.n_set, d)  # (L, C, n_set, D)

        # During evaluation, we leverage all (n_set) structures according to descriptions for modeling one category (N*C*n_set steps in total), 
        # instead of randomly picking one structure for each category (N*C steps in one epoch). 
        if not flag:
            high_prompt = high_prompt.unsqueeze(2).expand(-1, -1, self.n_set, -1, -1)
            high_prompt = torch.flatten(high_prompt, 1, 2)  # (L, C*n_set, n_set, D)

        high_prompt = high_prompt.permute(0, 2, 1, 3).type(self.dtype)
        x = (x + self.positional_embedding).type(self.dtype)
        x = x.permute(1, 0, 2)

        for layer_idx, layer in enumerate(self.transformer):
            if layer_idx > 0:
                prefix = x[:1]
                suffix = x[1 + self.n_tpro + self.n_set:]

                # global-level prompt
                ctx_g = global_prompt[layer_idx - 1].unsqueeze(1).expand(self.n_tpro, prefix.shape[1], -1)

                # high-level prompt
                ctx_h = high_prompt[layer_idx - 1]
                x = torch.cat([prefix, ctx_g, ctx_h, suffix], dim=0)

                # 'attn' is attention matrix from topological prompt learner, 
                # considering as low-level prompt which models relationships in an explicit way.
                x = layer(x, attn[:, layer_idx])
            elif layer_idx == 0:
                x = layer(x, attn[:, layer_idx])
            else:
                x = layer(x)

        x = x.permute(1, 0, 2)
        x = self.ln_final(x)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        if not flag:
            x = x.reshape(x.shape[0] // 5, 5, -1)

        return x

class PromptLearner(nn.Module):
    """path 2：global prompt learner for Hierachical text encoder"""
    def __init__(self, args, classnames, info_topo, clip_model):
        super().__init__()
        self.n_set = args.description_num  # number of descriptions for each category
        self.n_tpro = args.t_prompt_length  # prompt length
        self.dtype = clip_model.dtype
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.layers = len(clip_model.transformer.resblocks)

        # 初始化global prompt， global prompt for text encoder (except for the first layer)
        self.global_prompt = nn.ParameterList([nn.Parameter(torch.empty(self.n_tpro, self.ctx_dim).type(self.dtype))
                                               for _ in range(self.layers - 1)])
        for p in self.global_prompt:
            nn.init.normal_(p, std=0.02)

        # projector for learning high-level prompt (a.k.a p_ins)
        self.p_ins_projector = nn.Linear(self.ctx_dim, self.ctx_dim)

        # global prompt for the first layer of the text encoder
        self.p_input = nn.Parameter(torch.empty(self.n_tpro + self.n_set, self.ctx_dim))
        nn.init.normal_(self.p_input, std=0.02)

        self.classnames = [name.replace("_", " ") for name in classnames]
        self.info_topo = info_topo  # topological structure in a dictionary form
        self.n_cls = len(classnames)
        self.clip_model = clip_model

    def forward(self, feats, attns, flag):
        p_uni = self.global_prompt
        prompts, attn = [], []
        prompt_prefix = " ".join(["X"] * (self.n_tpro + self.n_set))

        if flag:
            for name in self.classnames:
                # 随机选择一个structure，而使用所有的class descriptions
                # For efficiency, we randomly pick one structure as a part of input during training,
                # while leveraging all descriptions of the category for learning high-level prompt.
                id = random.randint(0, self.n_set - 1)
                topo = self.info_topo[name][id]
                p = prompt_prefix + " " + name + ". " + ", ".join(topo['Entities']) + ". " + ", ".join(
                    topo['Attributes']) + "."
                attn.append(attns[name][:, id])
                prompts.append(p)
        else:
            for name in self.classnames:
                # We leverage all structures from descriptions as a part of input respectively during evaluation.
                for id in range(self.n_set):
                    topo = self.info_topo[name][id]
                    p = prompt_prefix + " " + name + ". " + ", ".join(topo['Entities']) + ". " + ", ".join(
                        topo['Attributes']) + "."
                    attn.append(attns[name][:, id])
                    prompts.append(p)

        attn = torch.stack(attn, dim=0)

        self.tokenized_prompts = torch.cat([clip.tokenize(p, truncate=True) for p in prompts]).cuda()  # (n_cls, n_tkn)
        with torch.no_grad():
            embedding = self.clip_model.token_embedding(self.tokenized_prompts).type(self.dtype)

        p_input = self.p_input.unsqueeze(0).expand(len(prompts), -1, -1)
        prefix = embedding[:, :1]
        suffix = embedding[:, 1 + self.n_tpro + self.n_set:]

        # the input of the prompted text encoder
        p_ori = torch.cat([prefix, p_input, suffix], dim=1)

        # generate corresponding high-level prompt (p_ins)
        p_ins = []
        (l, c, n, d) = feats.shape
        feats = feats.reshape(l, c * n, d)
        for idx in range(self.layers - 1):
            feat = feats[idx].float()
            feat = feat + self.p_ins_projector(feat)
            p_ins.append(feat)
        p_ins = torch.stack(p_ins, dim=0)

        return p_ori, p_ins, p_uni, attn

class TopoPromptLearner(nn.Module):
    """path 2：global prompt learner """
    def __init__(self, args, classnames, prompt_topo, clip_model):
        super().__init__()

        self.classnames = classnames
        self.dtype = clip_model.dtype
        self.n_set = args.description_num  # number of descriptions for each category
        self.n_tpro = args.t_prompt_length  # prompt length
        self.layers = len(clip_model.transformer.resblocks)

        # layer-wise scalar to weight indicating the strength of the relationship of entity-entity pairs and entity-attribute pairs
        self.e2e_scal = nn.Parameter(torch.zeros(self.layers, 1, 1, 1))
        self.e2a_scal = nn.Parameter(torch.zeros(self.layers, 1, 1, 1))

        self.attns_e2e = {classname: [] for classname in classnames}
        self.attns_e2a = {classname: [] for classname in classnames}

        prompt_prefix = " ".join(["X"] * (self.n_tpro + self.n_set))

        for classname in classnames:
            topos = prompt_topo[classname]
            for id in range(self.n_set):
                # generate text with classname, entities and attributes
                txt = self.generate_text(classname, prompt_prefix, topos[id])
                # 将text进行token化
                tokens = clip.tokenize(txt, truncate=True)[0]

                # generate pair-wise relationships
                e2e, e2a = self.extract_relationships(tokens, topos[id])

                # create attention matrix based on pair-wise relationships
                attn_e2e = self.create_attention_matrix(tokens, e2e)
                attn_e2a = self.create_attention_matrix(tokens, e2a)

                # save attention matrices
                self.attns_e2e[classname].append(attn_e2e)
                self.attns_e2a[classname].append(attn_e2a)

    # generate text with classname, entities and attributes
    def generate_text(self, classname, prompt_prefix, topo):
        entities = [w.lower() for w in topo['Entities']]
        attributes = [w.lower() for w in topo['Attributes']]
        txt = prompt_prefix + " " + classname + ". " + ", ".join(entities) + ". " + ", ".join(attributes) + "."
        return txt

    # generate pair-wise relationships from topological structure
    def extract_relationships(self, tokens, topo):
        entities = [w.lower() for w in topo['Entities']]
        attributes = [w.lower() for w in topo['Attributes']]
        e2e, e2a = [], []

        for w in topo['Entity-to-Entity Relationships']:
            if w['entity1'].lower() in entities and w['entity2'].lower() in entities:
                e1 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity1']))[0]))
                e2 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity2']))[0]))
                e2e.append([e1, e2])

        for w in topo['Entity-to-Attribute Relationships']:
            if w['entity'].lower() in entities and w['attribute'].lower() in attributes:
                e1 = list(self.align(tokens, self.truncate(clip.tokenize(w['entity']))[0]))
                e2 = list(self.align(tokens, self.truncate(clip.tokenize(w['attribute']))[0]))
                e2a.append([e1, e2])

        return e2e, e2a

    # create attention matrix based on pair-wise relationships
    def create_attention_matrix(self, tokens, relationships):
        n_tokens = len(tokens)
        attn = torch.zeros(n_tokens, n_tokens).cuda()

        for e in relationships:
            d11 = torch.tensor([[i] for i in e[0]]).type(torch.long)
            d21 = torch.tensor([e[1] for _ in range(len(e[0]))]).type(torch.long)
            d12 = torch.tensor([[i] for i in e[1]]).type(torch.long)
            d22 = torch.tensor([e[0] for _ in range(len(e[1]))]).type(torch.long)
            attn[d11, d21] += 1
            attn[d12, d22] += 1

        return attn

    # truncate token sequence according to EOS token
    def truncate(self, array):
        return array[:, 1:torch.argmax(array)]

    # find a sequence that matches the target token(s)
    def align(self, seq1, seq2):
        for idx in range(len(seq1) - len(seq2) + 1):
            if seq1[idx:idx + len(seq2)].equal(seq2):
                return range(idx, idx + len(seq2))
        return []

    def forward(self):
        attns = {}
        for classname in self.classnames:
            classname = classname.replace("_", " ")
            # weight generated matrices with two learnable scalars
            attns[classname] = self.e2e_scal * torch.stack(self.attns_e2e[classname]).cuda() + \
                               self.e2a_scal * torch.stack(self.attns_e2a[classname]).cuda()
        return attns

class CustomCLIP(nn.Module):
    def __init__(self, args, classnames, clip_model):
        super().__init__()
        for p in clip_model.parameters():
            p.requires_grad = False

        # 加载description和structure内容
        text_prompts = description.get_All_Descriptions(args)
        text_topos = structure.get_All_Structures(args)

        classnames = [name.replace("_", " ") for name in classnames]
        self.topo_prompt_learner = TopoPromptLearner(args, classnames, text_topos, clip_model)
        self.prompt_learner = PromptLearner(args, classnames, text_topos, clip_model)
        self.vision_prompt_learner = VisionPromptLearner(args, clip_model)

        self.image_encoder = VisionEncoder(args, clip_model)
        self.text_encoder = TextEncoder(args, clip_model)

        # self.model.to(torch.device("cuda"))
        self.text_encoder_zs = TextEncoderZS(args, clip_model)
        self.image_encoder_zs = VisionEncoderZS(args, clip_model)
        if torch.cuda.device_count() > 1:
            self.image_encoder = nn.DataParallel(self.image_encoder)
            self.text_encoder = nn.DataParallel(self.text_encoder)
            self.text_encoder_zs = nn.DataParallel(self.text_encoder_zs)
            self.image_encoder_zs = nn.DataParallel(self.image_encoder_zs)

        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.model = clip_model

        with torch.no_grad():
            # zs_feats: layer-wise class embeddings from frozen text encoder
            # zs_repres: final representations from frozen text encoder
            zs_feats, zs_repres = [], []
            for classname in classnames:
                texts = text_prompts[classname]
                texts = clip.tokenize(texts).cuda()
                class_embeddings, features = self.text_encoder_zs(texts)
                class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding /= class_embedding.norm()
                features /= features.norm(dim=-1, keepdim=True)
                zs_feats.append(features)
                zs_repres.append(class_embedding)
            #todo
            self.text_features_ft = torch.stack(zs_feats, dim=1).cuda()
            self.text_features_zs = torch.stack(zs_repres, dim=1).cuda()

    def forward(self, image):
        # 1. path 1
        # 1.1 prompt image encoder image feature
        x, p_visual = self.vision_prompt_learner(image)
        image_features = self.image_encoder(x, p_visual)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        # 1.2 frozen text encoder text feature
        text_features_zs = self.text_features_zs

        # 2. path 2
        # 2.1 frozen image encoder image feature
        image_features_zs = self.image_encoder_zs(image.type(self.dtype))
        image_features_zs = image_features_zs / image_features_zs.norm(dim=-1, keepdim=True)
        # 2.2 Hierachical text encoder text feature
        attns = self.topo_prompt_learner()
        p_ori, high_prompt, global_prompt, attns = self.prompt_learner(self.text_features_ft, attns, self.training)
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features = self.text_encoder(p_ori, high_prompt, global_prompt, tokenized_prompts, attns, self.training)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        # Since we use multiple structures for producing representations of one category,
        # we should take their mean value as the final representation.
        if not self.training:
            text_features = text_features.mean(dim=1)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # asymmetric loss
        #logits_i由prompt image encoder和frozen text encoder得到
        #logits_t由frozen image encoder和Hierachical prompt text encoder得到
        logit_scale = self.logit_scale.exp()
        logits_i = logit_scale * (image_features @ text_features_zs)
        logits_t = logit_scale * (image_features_zs @ text_features.t())
        logits = (logits_i + logits_t) / 2

        if self.training:
            return logits, logits_i, logits_t
        else:
            return logits