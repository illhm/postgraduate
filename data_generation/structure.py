import json
from openai import OpenAI
from tqdm import tqdm

global_api_key = "Ffy7Zo73qNffmZE9vxroEqv0uigsGdrkWO7oQREiJpzjpxoR" # Enter Your API Key!!!
dataset_name = "Caltech101" # Enter Your Dataset Name, e.g. EuroSAT

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

with open('D:/project/2024-AAAI-HPT-main/data/gpt_data/description/'+dataset_name+'.json', 'r') as f:
    prompt_templates = json.load(f)

#1.0.0版本之后的openai接口调用代码
client = OpenAI(
    api_key=global_api_key,
    base_url="https://api.chatanywhere.com.cn/v1"
)
result = {}
for classname in tqdm(prompt_templates.keys()):
    prompts = prompt_templates[classname]
    responses = [json.loads(get_completion(client, prompt)) for prompt in prompts]
    result[classname] = responses

    with open('D:/project/2024-AAAI-HPT-main/data/gpt_data/structure/'+dataset_name+'.json','w') as f:
        json.dump(result, f, indent=4)
