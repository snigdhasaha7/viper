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

prompt_file = 'rerank.prompt'
with open(f'prompts/{prompt_file}') as f:
    base_prompt = f.read().strip()

result_file = 'results_4.json'

with open(f'results/okvqav2/test/{result_file}', 'r') as f:
    top1data = json.load(f)

rerank_scores = []

for data in tqdm.tqdm(top1data):
    img = data['img_path']
    b64img = base64.b64encode(open(img, 'rb').read()).decode('ascii')
    query = data['query']
    answer = data['result'][2:-2]
    prompt = base_prompt.replace("INSERT_QUERY_HERE", query)\
                        .replace("INSERT_ANSWER_HERE", answer)

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
    #print(response.choices[0].message.content)
    try:
        thought, score = response.choices[0].message.content.split('\n')
        thought = thought[thought.index(' ') + 1:].strip()
        score = float(score[score.index(' ') + 1:].strip())
    except:
        print(response.choices[0].message.content)
        continue
    rerank_scores.append({
        'query': query,
        'answer': answer,
        'accuracy': data['accuracy'][0],
        'thought': thought,
        'score': score
    })

save_dir = 'rerank_results'
save_file = 'top1_rerank.json'
with open(f'{save_dir}/{save_file}', 'w') as f:
    json.dump(rerank_scores, f)

