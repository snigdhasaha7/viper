import json
from datasets.okvqa import OKVQADataset 
from torchvision import transforms
import ast

dataset = OKVQADataset(dataset_name='OKVQA', data_path='sample_okvqa', split='test', batch_size=1, max_samples=100, start_sample=0,\
                        image_transforms=transforms.Compose([transforms.ToTensor()]))


rerank_file = 'top5_rerank_12.json'
with open(f'rerank_results/{rerank_file}', 'r') as f:
    data = json.load(f) 

accuracy = 0
failure_modes = []
for d in data:
    reranked = d['ranked_answers']
    top_ans = reranked[0] 
    last_comma = top_ans.rfind(',') 
    top_ans = [[top_ans[1:last_comma]]]
    gt = [d['gt']]
    original_answers = d['answers']
    max_score = 0
    max_ans = ""
    all_0 = True
    for answer in original_answers:
        if answer == 'Execution Failure':
            continue 
        score = dataset.get_item_score(answer, gt[0])
        if score != 0:
            all_0 = False
        if score >= max_score:
            max_score = score
            max_ans = answer 

    if max_ans != top_ans[0][0]:
        failure_modes.append({
            'query': d['query'],
            'img_path': d['img_path'],
            'top_answer_gpt': top_ans[0][0],
            'ranked': reranked,
            'top_ans_eval': max_ans, 
            'top_ans_score': max_score,
            'all_0': all_0,
            'gt': gt
        })

    accuracy += dataset.accuracy(top_ans, gt)

print(f'Accuracy: {accuracy / len(data)}')
with open('rerank_results/top5_fail_12.json', 'w') as f:
    json.dump(failure_modes, f)