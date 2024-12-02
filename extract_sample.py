import json
import random

with open('results/okvqav2/test/results_12.json', 'r') as f:
    data = json.load(f)

new_data = []

for d in data:
    if not all(x == 0.0 for x in d['accuracy']):
        continue
    new_d = {
        'img_path': d['img_path'],
        'query': d['query'],
        'answer': d['answer']
    }
    new_data.append(new_d)

real_data = random.sample(new_data,20)

with open('extracted_samples.json', 'w') as f:
    json.dump(real_data, f)