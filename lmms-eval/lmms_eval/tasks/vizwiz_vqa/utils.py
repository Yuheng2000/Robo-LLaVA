import re
import os
import json
import yaml
import pathlib

import datetime
import statistics

from lmms_eval.tasks._task_utils.file_utils import generate_submission_file
from lmms_eval.tasks._task_utils.vqa_eval_metric import EvalAIAnswerProcessor

from loguru import logger as eval_logger

import jieba

import sacrebleu
# from sacrebleu.metrics import BLEU
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import PIL
import logging
import sys
import jieba

import requests
import base64
import re
def vizwiz_vqa_doc_to_visual(doc):
    return [doc["image"].convert("RGB")]

def metric_gpt4o(doc, pred_ans, answer):
    API_KEY = "869d966045f44db6ae0b8de02f7bf776"

    headers = {
        "Content-Type": "application/json",
        "api-key": API_KEY,
    }
    
    eval_prompt = """
        [Instruction]\nPlease act as an impartial judge and evaluate the quality 
        of the response provided by an AI assistant to
        the user question displayed below. Your evaluation should 
        consider correctness and helpfulness. You will be given
        a reference answer and the assistant’s answer. Begin 
        your evaluation by comparing the assistant’s answer with the
        reference answer. Identify and correct any mistakes. The 
        assistant has access to an image alongwith questions but
        you will not be given images. Therefore, please consider only 
        how the answer is close to the reference answer. If
        the assistant’s answer is not exactly same as or similar to 
        the answer, then he must be wrong. Be as objective as
        possible. Discourage uninformative answers. Also, 
        equally treat short and long answers and focus on the correctness
        of answers. After providing your explanation, you 
        must rate the response with either 0, 0.5 or 1 by strictly following
        this format: “[[rating]]”, for example: “Rating: [[0.5]]”.
        \n\n[Question]\n{question}\n\n[The Start of Reference
        Answer]\n{refanswer}\n[The End of Reference Answer]
        \n\n[The Start of Assistant’s Answer]\n{answer}\n[The
        End of Assistant’s Answer]
    """
    eval_texts = eval_prompt.format(question=doc["question"].strip(), refanswer=answer.strip(), answer=pred_ans)
    payload = {
    "messages": [
        {
        "role": "system",
        "content": [
            {
            "type": "text",
            "text": "You are an AI assistant that helps people find information."
            }
        ]
        },
        {
            "role": "user", 
            "content": eval_texts
        }
    ],
    "temperature": 0.7,
    "top_p": 0.95,
    "max_tokens": 800
    }

    ENDPOINT = "https://baai-emllm-eastus2.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise SystemExit(f"Failed to make the request. Error: {e}")
    response = response.json()
    content = response['choices'][0]['message']['content']
    pattern = r"Rating:\s*\[\[(\d+(\.\d+)?)\]\]"

    match = re.search(pattern, content)

    if match:
        rating_value = float(match.group(1))  # 提取第一个捕获组（数值部分）
    else:
        rating_value = -1.0

    return rating_value, content

def parse_pred_ans_NY(pred_ans):
    pred_label = None
    if pred_ans in ["yes", "no"]:
        pred_label = pred_ans
    else:
        prefix_pred_ans = pred_ans[:4]

        if "yes" in prefix_pred_ans:
            pred_label = "yes"
        elif "no" in prefix_pred_ans:
            pred_label = "no"
        else:
            pred_label = "other"
    return pred_label

def vizwiz_vqa_process_results(doc, result):
    eval_ai_processor = EvalAIAnswerProcessor()
    assert len(result) == 1, f"The result should be a list of length 1, but got {len(result)}."
    resAns = eval_ai_processor(result[0])
    accuracy = 0

    if "answers" in doc and doc["answers"] is not None:
        gtAcc = []

        for i in range(len(doc["answers"])):
            doc["answers"][i] = eval_ai_processor(doc["answers"][i])

        for i in range(len(doc["answers"])):
            otherGTAns = [doc["answers"][j] for j in range(len(doc["answers"])) if i != j]
            matchingAns = [item for item in otherGTAns if item == resAns]
            acc = min(1, float(len(matchingAns)) / 3)
            gtAcc.append(acc)
        if gtAcc:
            accuracy = statistics.mean(gtAcc)
        else:
            accuracy = 0

    if "answers" in doc:
        scores = []
        gpt_scores = []
        for answer in doc['answers'][:2]:
            pred = result[0].replace('A:',' ').replace(':',' ').replace('(',' ').replace(')',' ').strip()
            pred = pred.replace('\n', "").lower()
            gt_ans = answer.strip()
            # parser
            doc['question_field']='waiting_to_sure'
            if gt_ans in ['yes', 'no']:
                pred_ans = parse_pred_ans_NY(pred)
            else:
                pred_ans = pred
            
            #gt_ans = doc["answer"].lower()
            pred_ans = pred_ans.lower()
            # ans =  pred_ans
            # gt = gt_ans
            ans = "".join(jieba.cut(pred_ans)).strip()
            gt = ''.join(jieba.cut(gt_ans)).strip()
                
            score =[
                max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(1,0,0,0)) for lenth in range(-5,5)),
                max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,1,0,0)) for lenth in range(-5,5)),
                max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,1,0)) for lenth in range(-5,5)),
                max(sentence_bleu([ans[:len(gt)-lenth].split()], gt.split(), smoothing_function=SmoothingFunction().method4, weights=(0,0,0,1)) for lenth in range(-5,5)),
            ]
            gpt_score = - 1.0 
            try:
                gpt_score, _ = metric_gpt4o(doc, pred_ans, answer)
            except:
                max_try = 2
                while(max_try>0):
                    try: 
                        gpt_score, _ = metric_gpt4o(doc, pred_ans, answer)
                        break
                    except:
                        continue
                    max_try -= 1
            if gpt_score == -1.0:
                gpt_score = 0.0
            gpt_scores.append(gpt_score)
            scores.append(score)
        gpt_score = sum(gpt_scores) / len(gpt_scores)
        score = [[sc[0]for sc in scores], [sc[1]for sc in scores],[sc[2]for sc in scores],[sc[3]for sc in scores]]
        score = [sum(sc) / len(sc) for sc in score]

    return {"vizwiz_vqa":{"exact_match": accuracy, "ans": ans , "gt": gt,
                       "BLEU_1": score[0], "BLEU_2": score[1], "BLEU_3": score[2], "BLEU_4": score[3], "GPT-Score": gpt_score}}

    
    # return {
    #     "exact_match": accuracy,
    #     "submission": {
    #         "image": f"{doc['question_id']}.jpg",
    #         "answer": resAns,
    #     },
    # }


def vizwiz_vqa_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    text = f"{pre_prompt}{doc['question'].capitalize()}{post_prompt}"
    return text


def vizwiz_vqa_aggregate_submissions(results, args):
    now_date_time = datetime.datetime.now().strftime("%Y-%m%d-%H%M-%S")
    submission_file_name = f"vizwiz_vqa-test-submission-{now_date_time}.json"
    path = generate_submission_file(submission_file_name, args)
    with open(path, "w") as f:
        json.dump(results, f)
    print(f"Submission file saved to {path}")
