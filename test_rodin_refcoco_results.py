import torch
import numpy as np
from som_mask_overlay.visualizer import Visualizer
from PIL import Image

import openai
from openai import AzureOpenAI
import base64
from io import BytesIO
import re
import json

import ipdb
st = ipdb.set_trace

def extract_float(string):
    # Find the first occurrence of a floating-point number in the string
    match = re.search(r"\d*\.\d+|\d+", string)
    if match:
        # Convert the matched string to a float
        return float(match.group(0))
    else:
        # Return None or handle cases where no valid float is found
        return None

def inference_sam_m2m_interactive(image, mask, label, label_mode='1', alpha=0.4, anno_mode=['Mask', 'Mark', 'Box']):
    visual = Visualizer(np.asarray(image))
    mask = mask
    demo = visual.draw_binary_mask_with_number(mask, text=str(label), label_mode=label_mode, alpha=alpha, anno_mode=anno_mode, color='violet')
    im = demo.get_image()
    return im

def main(res_path, image_root_path):
    with open('api.key') as f:
        openai.api_key = f.read().strip()

    client = AzureOpenAI(
        api_key = openai.api_key,
        api_version = "2023-05-15",
        azure_endpoint = 'https://katfgroup-gpt4-ce.openai.azure.com/'
    )

    prompt_file = 'rerank_detection.prompt'
    with open(f'prompts/{prompt_file}') as f:
        base_prompt = f.read().strip()

    res = torch.load(res_path)
    test_restuls = []
    cnt = 0
    for x in res:
        print(x['text_caption'])
        image = Image.open(image_root_path + x['input_image'].split('/')[-1])
        scores = []
        thoughts = []

        prompt = base_prompt.replace("INSERT_DESCRIPTION_HERE", x['text_caption'])
        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt
                },
            ]
        }]

        for i in range(len(x['top_5_pred_masks'])):
            mask = x['top_5_pred_masks'][i].numpy()
            if mask.sum() == 0:
                scores.append(0)
                continue
            im = inference_sam_m2m_interactive(image, mask, i+1)
            masked_image = Image.fromarray(im.astype('uint8'), 'RGB')
            buffered = BytesIO()
            masked_image.save(f"/home/yuzhouwa/refcoco_results/gpt_score/mask_{i}.jpg", format="JPEG")
            masked_image.save(buffered, format="JPEG")
            b64_masked_image = base64.b64encode(buffered.getvalue()).decode('ascii')

            messages[0]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{b64_masked_image}",
                }
            })
            
        try:
            response = client.chat.completions.create(
                model = 'gpt-4o',
                messages = messages,
                n = 1
            )
        except:
            client = AzureOpenAI(
                api_key = openai.api_key,
                api_version = "2023-05-15",
                azure_endpoint = 'https://katfgroup-gpt4-ce.openai.azure.com/'
            )
            break

        try:
            tmp = response.choices[0].message.content.split('Score')
            # for substr in tmp:
            #     if 'score' in substr.lower():
            #         score = substr
            score = tmp[-1]
            score = score[score.index(' ') + 1:].strip().split(';')
            score = [extract_float(x) for x in score]
            if score == None:
                score = 0
            scores = score
            thoughts.append({
                'output': response.choices[0].message.content,
                'score': score
            })
        except:
            # st()
            # print(response.choices[0].message.content)
            scores.append(0)
            continue

        with open(f'/home/yuzhouwa/refcoco_results/gpt_score/thoughts_{cnt}.json', 'w') as f:
            json.dump(thoughts, f)

        if len(scores) == 5:
            ious = []
            for pred_mask in x['top_5_pred_masks']:
                intersection = (pred_mask & x['gt_mask']).float().sum()
                union = (pred_mask | x['gt_mask']).float().sum()
                iou = (intersection / union).item() if union > 0 else 0.0
                ious.append(iou)
            ious = np.array(ious)

            top_gpt_score = max(scores)
            gpt_top_1_id = scores.index(top_gpt_score)
            gpt_top_5_id = [0, 1, 2, 3, 4]
            # print(ious)
            # print(scores)
            # print(f'top 1 model iou: {ious[0]}; top 5 model iou: {ious[:5].max()}; top 1 gpt iou: {ious[gpt_top_1_id]}; top 5 gpt iou: {ious[gpt_top_5_id].max()}')
            with open(f'/home/yuzhouwa/refcoco_results/gpt_eval_result_jsons/result_{cnt}.json', 'w') as f:
                json.dump({
                    'caption': x['text_caption'],
                    'gpt_scores': scores,
                    'ious': ious.tolist(),
                    'top 1 model iou': ious[0].tolist(),
                    'top 5 model iou:': ious[:5].max().tolist(),
                    'top 1 gpt iou': (ious[gpt_top_1_id]).tolist(),
                    'top 5 gpt iou': (ious[gpt_top_5_id].max()).tolist(),
                    'top 1 model acc': int(ious[0] > 0.5),
                    'top 5 model acc': int(ious[:5].max() > 0.5),
                    'top 1 gpt acc': int(ious[gpt_top_1_id] > 0.5),
                    'top 5 gpt acc': int(ious[gpt_top_5_id].max() > 0.5),
                }, f)
        cnt += 1
        print(cnt)

if __name__ == '__main__':
    # change the following
    main(
        res_path = '/home/yuzhouwa/refcoco_results/refcoco_first_100_results.pth',
        image_root_path = '/home/yuzhouwa/refcoco_results/images/'
    )
