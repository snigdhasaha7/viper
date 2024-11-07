import openai
from openai import AzureOpenAI
import base64
import json
import tqdm
import ast
import re

with open('api.key') as f:
    openai.api_key = f.read().strip()

client = AzureOpenAI(
    api_key = openai.api_key,
    api_version = "2023-05-15",
    azure_endpoint = 'https://katfgroup-gpt4-ce.openai.azure.com/'
)

prompt_file = 'rerank_top5.prompt'
with open(f'prompts/{prompt_file}') as f:
    base_prompt = f.read().strip()

result_file = 'results_5.json'

with open(f'results/okvqav2/test/{result_file}', 'r') as f:
    top1data = json.load(f)

rerank_scores = []

for data in tqdm.tqdm(top1data):
    img = data['img_path']
    b64img = base64.b64encode(open(img, 'rb').read()).decode('ascii')
    query = data['query']
    answers = ast.literal_eval(data['result'])
    for i in range(5):
        if answers[i] == None:
            answers[i] = "Execution Failure"
    prompt = base_prompt.replace("INSERT_QUERY_HERE", query)\
                        .replace("INSERT_ANSWER1_HERE", answers[0])\
                        .replace("INSERT_ANSWER2_HERE", answers[1])\
                        .replace("INSERT_ANSWER3_HERE", answers[2])\
                        .replace("INSERT_ANSWER4_HERE", answers[3])\
                        .replace("INSERT_ANSWER5_HERE", answers[4])\

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

    try:
        thoughts_match = re.search(r'Thoughts: (.*?)(?=\n\nReranked_answers:)', text, re.DOTALL)
        thought = thoughts_match.group(1).strip() if thoughts_match else ''

        ans1_ind = text.index('1. ')
        ans2_ind = text.index('2. ')
        ans3_ind = text.index('3. ')
        ans4_ind = text.index('4. ')
        ans5_ind = text.index('5. ')
        ans1 = text[ans1_ind+3:ans2_ind].strip()
        ans2 = text[ans2_ind+3:ans3_ind].strip()
        ans3 = text[ans3_ind+3:ans4_ind].strip()
        ans4 = text[ans4_ind+3:ans5_ind].strip()
        ans5 = text[ans5_ind+3:].strip()
        reranked_answers = [ans1, ans2, ans3, ans4, ans5]
        
    except:
        print(text)
        continue

    rerank_scores.append({
        'query': query,
        'answers': answers,
        'gt': data['answer'],
        'img_path': img,
        'accuracy': data['accuracy'][0],
        'thought': thought,
        'ranked_answers': reranked_answers
    })

save_dir = 'rerank_results'
save_file = 'top5_rerank.json'
with open(f'{save_dir}/{save_file}', 'w') as f:
    json.dump(rerank_scores, f)

