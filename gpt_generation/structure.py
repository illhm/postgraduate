import json
from openai import OpenAI
from tqdm import tqdm
import os.path as osp

global_api_key = "Ffy7Zo73qNffmZE9vxroEqv0uigsGdrkWO7oQREiJpzjpxoR" # Enter Your API Key!!!
#class structures缓存
structures=None

def get_completion(client, prompt, model="gpt-3.5-turbo", temperature=1):
    instruction = "Please reconstruct the following sentence into 4 parts and output a JSON object. \
              1. All entities, \
              2. All attributes, \
              3. Relationships between entity and entity, \
              4. Relationships between attributes and entities. \
              The target JSON object contains the following keys: \
              Entities, \
              Attributes, \
              Entity-to-Entity Relationships, \
              Entity-to-Attribute Relationships. \
              For the key Entity-to-Entity Relationships, the value is a list of JSON object that contains the following keys: entity1, relationship, entity2. \
              For the key Entity-to-Attribute Relationships, the value is a list of JSON object that contains the following keys: entity, relationship, attribute. \
              Do not output anything other than the JSON object. \
              \
              Sentence: '''{}'''".format(prompt)
    messages = [{"role": "system", "content": "You are good at image classification."}, {"role": "user", "content": instruction}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

def get_All_Structures(args):
    # 要使用全局变量并赋值，需要使用global关键字
    global structures
    # 1. 若没有缓存
    if structures is None:
        path = osp.join(args.gpt_dir, 'structure', args.dataset + '.json')
        # 1.1 若文件存在，则直接读取
        if osp.isfile(path):
            with open(path, 'r') as f:
                structures = json.load(f)
        # 1.2 若文件不存在，则调用gpt生成structures
        else:
            # 1.2.1 ——1.0.0版本之后的openai接口调用代码
            client = OpenAI(
                api_key=global_api_key,
                base_url="https://api.chatanywhere.com.cn/v1"
            )
            # 1.2.2 先读取description，作为prompt
            descrip_path = osp.join(args.gpt_dir, 'description', args.dataset + '.json')
            with open(descrip_path, 'r') as f:
                prompt_templates = json.load(f)

            structures = {}
            structure_path = osp.join(args.gpt_dir, 'structure', args.dataset + '.json')
            # 1.2.3 对于所有class，逐个根据description生成structure
            for classname in tqdm(prompt_templates.keys()):
                prompts = prompt_templates[classname]
                responses = [json.loads(get_completion(client, prompt)) for prompt in prompts]
                structures[classname] = responses
            # 1.2.4 最后将缓存写入文件
            with open(structure_path, 'w') as f:
                json.dump(structures, f, indent=4)
    return structures

def get_Classes_Structures(args, classnames):
    structures = get_All_Structures(args)
    # 2. 从缓存中读取对应classes的structures
    reuslt={i:structures[i] for i in classnames}
    return reuslt


