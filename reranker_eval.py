import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import json

rerank_file = 'top1_rerank.json'
with open(f'rerank_results/{rerank_file}', 'r') as f:
    data = json.load(f) 

accuracy = 0
y_true = [] 
y_score = []
for d in data:
    acc = d['accuracy']
    if acc >= 0.5:
        acc = 1.0
    y_true.append(acc)
    y_score.append(d['score'] / 10)
    accuracy += (d['accuracy'] == 1 and d['score'] >= 5) or (d['accuracy'] == 0 and d['score'] < 5)

print(f'Accuracy of Verifier: {accuracy / len(y_true)}')

precision, recall, thresholds = precision_recall_curve(y_true, y_score)


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='b', label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig('reranker_pr.png')