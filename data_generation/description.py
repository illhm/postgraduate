import json
from openai import OpenAI
from tqdm import tqdm

global_api_key = "sk-Ffy7Zo73qNffmZE9vxroEqv0uigsGdrkWO7oQREiJpzjpxoR" # Enter Your API Key!!!
dataset_name = "Caltech101" # Enter Your Dataset Name, e.g. EuroSAT

templates = ["What does {} look like among all {}? ",
             "What are the distinct features of {} for recognition among all {}? ",
             "How can you identify {} in appearance among all {}? ",
             "What are the differences between {} and other {} in appearance? ",
             "What visual cue is unique to {} among all {}? "]

infos = {
    'ImageNet':             ["{}",                "objects"],
    'OxfordPets':           ["a pet {}",          "types of pets"], 
    'Caltech101':           ["{}",                "objects"],
    'DescribableTextures':  ["a {} texture",      "types of texture"],
    'EuroSAT':              ["{}",                "types of land in a centered satellite photo"],
    'FGVCAircraft':         ["a {} aircraft",     "types of aircraft"],
    'Food101':              ["{}",                "types of food"],
    'OxfordFlowers':        ["a flower {}",       "types of flowers"],
    'StanfordCars':         ["a {} car",          "types of car"],
    'SUN397':               ["a {} scene",        "types of scenes"],
    'UCF101':               ["a person doing {}", "types of action"],
}

def get_completion(client, prompt, model="gpt-3.5-turbo", temperature=1):
    """
    调用openai的api接口，给出text提示，返回description
    Args:
        client:
        prompt:
        model:
        temperature:
    Returns:
    """
    messages = [{"role": "system", "content": "You are good at image classification."}, {"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()

#1.0.0版本之后的openai接口调用代码
client = OpenAI(
    api_key=global_api_key,
    base_url="https://api.chatanywhere.com.cn/v1"
)

with open('D:/project/2024-AAAI-HPT-main/data/gpt_data/classname/'+dataset_name+'.txt', 'r') as f:
    classnames = f.read().split("\n")[:-1]
result = {}

for classname in tqdm(classnames):
    info = infos[dataset_name]
    # prompt替换和生成
    prompts = [template.format(info[0], info[1]).format(classname) + "Describe it in 20 words." for template in templates]
    print("\r\n prompts:{}".format(prompts))
    responses = [get_completion(client,prompt) for prompt in prompts]
    result[classname] = responses

    with open('D:/project/2024-AAAI-HPT-main/data/gpt_data/description/'+dataset_name+'.json','w') as f:
        json.dump(result, f, indent=4)


def get_description_from_classes(classenames):
    for classname in tqdm(classnames):
        info = infos[dataset_name]
        # prompt替换和生成
        prompts = [template.format(info[0], info[1]).format(classname) + "Describe it in 20 words." for template in
                   templates]
        print("\r\n prompts:{}".format(prompts))
        responses = [get_completion(client, prompt) for prompt in prompts]
        result[classname] = responses

        with open('D:/project/2024-AAAI-HPT-main/data/gpt_data/description/' + dataset_name + '.json', 'w') as f:
            json.dump(result, f, indent=4)