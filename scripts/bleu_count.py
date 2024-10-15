import json
import os
import jsonlines

def read_jsonl(file):
    results = []
    with open(file, "r", encoding='utf-8') as f:
        for item in jsonlines.Reader(f):
            results.append(item)
    return results


def read_json(file):
    with open(file, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_json(file, data):
    with open(file, "w") as f:
        json.dump(data, f, indent=2)
        

# 需要修改路径
data =  read_json("/home/jiyuheng/Robo-LLaVA/scripts/logs/0919_1827_llava_onevision_llava_onevision_model_args_2df378/robovqa.json")

# 初始化总和变量
total_bleu_1 = 0
total_bleu_2 = 0
total_bleu_3 = 0
total_bleu_4 = 0

# 日志数量
count = len(data['logs'])

# 计算每个 BLEU 的总和
for log in data['logs']:
    total_bleu_1 += log['robovqa']['BLEU_1']
    total_bleu_2 += log['robovqa']['BLEU_2']
    total_bleu_3 += log['robovqa']['BLEU_3']
    total_bleu_4 += log['robovqa']['BLEU_4']
    
# 计算均值
mean_bleu_1 = total_bleu_1 / count
mean_bleu_2 = total_bleu_2 / count
mean_bleu_3 = total_bleu_3 / count
mean_bleu_4 = total_bleu_4 / count

print(f"BLEU_1 Mean: {mean_bleu_1}")
print(f"BLEU_2 Mean: {mean_bleu_2}")
print(f"BLEU_3 Mean: {mean_bleu_3}")
print(f"BLEU_4 Mean: {mean_bleu_4}")