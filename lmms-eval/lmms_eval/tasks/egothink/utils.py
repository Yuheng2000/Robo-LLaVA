from collections import defaultdict
import os
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
eval_logger = logging.getLogger("lmms-eval")
dir_name = os.path.dirname(os.path.abspath(__file__))


def metric_gpt4o(doc, pred_ans):
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
    eval_texts = eval_prompt.format(question=doc["question"].strip(), refanswer=doc["answer"].strip(), answer=pred_ans)
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

# def metric_gpt4o(doc, pred_ans):
#     API_KEY = "869d966045f44db6ae0b8de02f7bf776"
#     # IMAGE_PATH = "YOUR_IMAGE_PATH"
#     # encoded_image = base64.b64encode(open(IMAGE_PATH, 'rb').read()).decode('ascii')
#     headers = {
#         "Content-Type": "application/json",
#         "api-key": API_KEY,
#     }
#     eval_prompt = """
#         [Instruction]\nPlease act as an impartial judge and evaluate the quality 
#         of the response provided by an AI assistant to
#         the user question displayed below. Your evaluation should 
#         consider correctness and helpfulness. You will be given
#         a reference answer and the assistant’s answer. Begin 
#         your evaluation by comparing the assistant’s answer with the
#         reference answer. Identify and correct any mistakes. The 
#         assistant has access to an image alongwith questions but
#         you will not be given images. Therefore, please consider only 
#         how the answer is close to the reference answer. If
#         the assistant’s answer is not exactly same as or similar to 
#         the answer, then he must be wrong. Be as objective as
#         possible. Discourage uninformative answers. Also, 
#         equally treat short and long answers and focus on the correctness
#         of answers. After providing your explanation, you 
#         must rate the response with either 0, 0.5 or 1 by strictly following
#         this format: “[[rating]]”, for example: “Rating: [[0.5]]”.
#         \n\n[Question]\n{question}\n\n[The Start of Reference
#         Answer]\n{refanswer}\n[The End of Reference Answer]
#         \n\n[The Start of Assistant’s Answer]\n{answer}\n[The
#         End of Assistant’s Answer]
#     """
#     # Payload for the request
#     payload = {
#     "messages": [
#         {
#         "role": "system",
#         "content": [
#             {
#             "type": "text",
#             "text": "You are an AI assistant that helps people find information."
#             }
#         ]
#         },
#         {
#             "role": "user", 
#             "content": eval_prompt.format("question"=doc["question"].strip(), "refanswer"=doc["answer"].strip(), "answer"=pred_ans)
#         }
#     ],
#     "temperature": 0.7,
#     "top_p": 0.95,
#     "max_tokens": 800
#     }

#     ENDPOINT = "https://baai-emllm-eastus2.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

#     # Send request
#     try:
#         response = requests.post(ENDPOINT, headers=headers, json=payload)
#         response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
#     except requests.RequestException as e:
#         raise SystemExit(f"Failed to make the request. Error: {e}")

#     return response.json()


def egothink_doc_to_images(doc):

    # import pdb
    # pdb.set_trace()
    image=doc['image']
    image_list=[image]
    return image_list
    # doc['image'].save('new_example.png')
    # visual_path='/home/henry/LLaVA-NeXT/new_example.png'
    # return [visual_path]

    # else :
    #     raise ValueError("format is wrong ")

def egothink_doc_to_text(doc, lmms_eval_specific_kwargs=None):
                                                              
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    
    pre_prompt = ""
    post_prompt = ""
    if "pre_prompt" in lmms_eval_specific_kwargs:
        pre_prompt = lmms_eval_specific_kwargs["pre_prompt"]
    if "post_prompt" in lmms_eval_specific_kwargs:
        post_prompt = lmms_eval_specific_kwargs["post_prompt"]
    question = doc["question"].strip()
    return f"{pre_prompt}{question}{post_prompt}"


def egothink_doc_to_target(doc):
    answer = doc["answer"]
    return answer



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


def parse_pred_ans_choice(pred_ans):
    return pred_ans.replace(" ", "")[0]


def egothink_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case mme score), value: metric value
    """
    # import pdb
    # pdb.set_trace()
    pred = results[0].strip()
    pred = pred.replace('\n', "").lower()
    # parser
    doc['question_field']='waiting_to_sure'
    if doc["question_field"] == "N/Y":
        pred_ans = parse_pred_ans_NY(pred)
    elif doc["question_field"] == "Choices":
        pred_ans = parse_pred_ans_choice(pred)
    else:
        pred_ans = pred
    
    #gt_ans = doc["answer"].lower()
    pred_ans = pred_ans.lower()
    gt_ans = doc["answer"].strip()
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
    
    gpt_score, _ = metric_gpt4o(doc, pred_ans)
    # print(gpt_score)
    # score
    # score = 1 if (doc["question_field"] == "Q/A" and anls_score(prediction=pred_ans, gold_labels=[gt_ans], threshold=0.95) >= 0.4) \
    #                 or (gt_ans == pred_ans) \
            # else 0
    return {"egothink":{ "ans": ans , "gt": gt,
                       "BLEU_1": score[0], "BLEU_2": score[1], "BLEU_3": score[2], "BLEU_4": score[3], "GPT-Score": gpt_score}}
    # return {"egothink":{ "ans": ans , "gt": gt,
    #                    "BLEU_1": score[0], "BLEU_2": score[1], "BLEU_3": score[2], "BLEU_4": score[3]}}

def egothink_aggregate_results(results):
    """
    Args:
        results: a list of values returned by process_results
    Returns:
        A score
    """
    summary = defaultdict(dict)
    BLEU_1 = 0
    BLEU_2 = 0
    BLEU_3 = 0
    BLEU_4 = 0
    
    for result in results:
        video_id = result["video_id"]
        BLEU_1 += result["BLEU_1"]
        BLEU_2 += result["BLEU_2"]
        BLEU_3 += result["BLEU_3"]
        BLEU_4 += result["BLEU_4"]

    BLEU_1 = BLEU_1 / (len(results))
    BLEU_2 = BLEU_2 / (len(results))
    BLEU_3 = BLEU_3 / (len(results))
    BLEU_4 = BLEU_4 / (len(results))
        

    print(f"BLEU-1: {BLEU_1:.2f},  BLEU-2: {BLEU_2:.2f},  BLEU-3: {BLEU_3:.2f},  BLEU-4: {BLEU_4:.2f}")
    # cnt_con = cnt_con / (len(results) / 3)
    eval_logger.info(f"BLEU-1: {BLEU_1:.2f},  BLEU-2: {BLEU_2:.2f},  BLEU-3: {BLEU_3:.2f},  BLEU-4: {BLEU_4:.2f}")
    return BLEU_1, BLEU_2, BLEU_3, BLEU_4


def egothink_aggregate_res_bleu_1(results):
    BLEU_1, _, _ , _ = egothink_aggregate_results(results)
    return BLEU_1

def egothink_aggregate_res_bleu_2(results):
    _, BLEU_2, _ , _ = egothink_aggregate_results(results)
    return BLEU_2
def egothink_aggregate_res_bleu_3(results):
    _, _, BLEU_3 , _ = egothink_aggregate_results(results)
    return BLEU_3
def egothink_aggregate_res_bleu_4(results):
    _, _, _ , BLEU_4 = egothink_aggregate_results(results)
    return BLEU_4

    

# def egothink_aggregate_results(results):
#     """
#     Args:
#         results: a list of values returned by process_results
#     Returns:
#         A score
#     """
#     summary = defaultdict(dict)
#     for result in results:
#         video_id = result["video_id"]
#         score = result["score"]
#         if video_id not in summary.keys():
#             summary[video_id] = 0
#         summary[video_id] += score

#     cnt_con = 0
#     for video_id, score in summary.items():
#         if score == 3:
#             cnt_con += 1
    
#     print("Consistency Cases are ", cnt_con)
#     cnt_con = cnt_con / (len(results) / 3)
#     eval_logger.info(f"ConScore_D: {cnt_con:.2f}")
#     return cnt_con