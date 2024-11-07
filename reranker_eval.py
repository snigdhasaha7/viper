import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
import json

rerank_file = 'top1_rerank.json'
with open(f'rerank_results/{rerank_file}', 'r') as f:
    data = json.load(f) 

accuracy = 0
y_true = [] 
y_score = []
y_threshold_score = []
for d in data:
    acc = d['accuracy']
    if acc >= 0.5:
        acc = 1.0
    y_true.append(acc)
    y_score.append(d['score'] / 10)
    acc_score = (acc == 1 and d['score'] >= 5) or (acc == 0 and d['score'] < 5)
    if acc_score != 1: 
        print(d)
    y_threshold_score.append(acc_score)
    accuracy += acc_score

print(f'Accuracy of Verifier: {accuracy / len(y_true)}')

precision, recall, thresholds = precision_recall_curve(y_true, y_score)
prec = precision_score(y_true, y_threshold_score)
rec = recall_score(y_true, y_threshold_score)

print(f'Precision: {prec}')
print(f'Recall: {rec}')
print(f'F1: {2 * (prec * rec / (prec + rec))}')


plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.', color='b', label='PR Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid()
plt.savefig('reranker_pr.png')