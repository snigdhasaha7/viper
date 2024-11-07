from datasets.okvqa import OKVQADataset 
from torchvision import transforms
import json
import ast

dataset = OKVQADataset(dataset_name='OKVQA', data_path='sample_okvqa', split='test', batch_size=1, max_samples=100, start_sample=0,\
                        image_transforms=transforms.Compose([transforms.ToTensor()]))

with open('results/okvqav2/test/results_5.json', 'r') as f:
    data = json.load(f)

accuracy = 0

for d in data:
    pred = [ast.literal_eval(d['result'])]
    gt = [d['answer']]
    accuracy += dataset.accuracy(pred, gt)

print(f'Accuracy: {accuracy/len(data)}')