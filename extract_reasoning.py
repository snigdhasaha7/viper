import openai
from openai import AzureOpenAI
import base64
import json
import tqdm

with open('api.key') as f:
    openai.api_key = f.read().strip()

client = AzureOpenAI(
    api_key = openai.api_key,
    api_version = "2023-05-15",
    azure_endpoint = 'https://katfgroup-gpt4-ce.openai.azure.com/'
)

with open('extracted_samples.json', 'r') as f:
    sample_data = json.load(f)

prompt_file = 'reasoning.prompt'
with open(f'prompts/{prompt_file}') as f:
    base_prompt = f.read().strip()

output = [] 

for data in tqdm.tqdm(sample_data):
    img = data['img_path']
    b64img = base64.b64encode(open(img, 'rb').read()).decode('ascii')
    query = data['query']
    prompt = base_prompt.replace("INSERT_QUERY_HERE", query)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64img}",
                    }
                }
            ]
        }]

    response = client.chat.completions.create(
            model='gpt-4o',
            messages=messages,
            n=1
        )
    text = response.choices[0].message.content
    # try:
    #     thought, answer = text.split('\n')
    #     thought = thought[thought.index(' ') + 1:].strip()
    #     answer = answer[answer.index(' ') + 1:].strip()
    # except:
    #     print(text)

    output.append({
        'img_path': img, 
        'query': query, 
        'gt_answer': data['answer'], 
        'result': text
    })


with open('extracted_thoughts.json', 'w') as f:
    json.dump(output, f)