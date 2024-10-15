# import json
# import re
# import copy
# import transformers


# with open("/home/dataset_zoo/M4-Instruct/m4_instruct_annotations.json", "r") as json_file:
#     cur_data_dict = json.load(json_file)

DEFAULT_IMAGE_TOKEN = "<image>"

# def preprocess_multimodal(sources, data_args=None):
#     is_multimodal = True
#     for source in sources:
#         for sentence in source:
#             import pdb; pdb.set_trace()
#             num_im = len(re.findall(DEFAULT_IMAGE_TOKEN, sentence["value"]))
#             # 只有一个图像，把他放到最前面
#             if num_im == 1 and DEFAULT_IMAGE_TOKEN in sentence["value"] and not sentence["value"].startswith(DEFAULT_IMAGE_TOKEN):
#                 sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
#                 sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
#                 sentence["value"] = sentence["value"].strip()
#                 # if "mmtag" in conversation_lib.default_conversation.version:
#                 #     sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>")
#             replace_token = DEFAULT_IMAGE_TOKEN
#             # mm_use_im_start_end用来在<image>前后加<im_start>和<im_end>
#             # if data_args.mm_use_im_start_end:
#             #     replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
#             sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

#             # For videoInstruct-100k noisy_data. TODO: Ask Yuanhan to clean the data instead of leaving the noise code here.
#             sentence["value"] = sentence["value"].replace("QA_GT_caption_based_noisy", "")

#     return sources



# print(f"Loaded {len(cur_data_dict)} samples from /home/dataset_zoo/M4-Instruct/m4_instruct_annotations.json")
# x = [cur_data_dict[0]]
# sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in x]))
# # [[{'from': 'human', 'value': "<image><image>\nWhat's the detailed difference between the 2 images? Please list in detail."}, {'from': 'gpt', 'value': 'The differences between the two images are:\n\n1. In the second image, there are leaves falling from the sunflowers and the surrounding foliage.\n2. The ground in the second image is covered with a layer of fallen leaves, adding a carpet-like appearance.'}]]

# data_dict = preprocess_qwen(sources, tokenizer, has_image=True)




# sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in x]), self.data_args)


# import re

# def sample_and_replace(text, n, img_token):
#     # 找到所有img的位置
#     matches = [match.start() for match in re.finditer(img_token, text)]
    
#     # 如果数量小于或等于 n，不做修改
#     if len(matches) <= n:
#         return text
    
#     # 计算采样间隔
#     step = len(matches) // n
    
#     # 选择保留的索引
#     sampled_indices = set(matches[i * step] for i in range(n))
    
#     # 替换未采样到的 "SPECIAL" 为 ""
#     result = []
#     last_end = 0
    
#     for start in matches:
#         end = start + len(img_token)
#         # 只保留采样到的img_token
#         if start in sampled_indices:
#             result.append(text[last_end:end])
#             last_end = end
#         else:
#             # 跳过img_token并删除前面的空格
#             result.append(text[last_end:start].rstrip())
#             last_end = end
    
#     # 添加剩余部分
#     result.append(text[last_end:])
    
#     return ''.join(result)

# # 示例
# text = "This is <image><image><image><image><image> a test string with multiple markers <image><image><image><image><image>."
# n = 2
# result = sample_and_replace(text, n, DEFAULT_IMAGE_TOKEN)
# print(result)
# import re
# grid_pinpoints = "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]"
# matches = re.findall(r"\((\d+)x(\d+)\)", grid_pinpoints)
# range_start = tuple(map(int, matches[0]))
# range_end = tuple(map(int, matches[-1]))

# range_start = (1,1)
# range_end = (6,4)
# grid_pinpoints = [(i, j) for i in range(range_start[0], range_end[0] + 1) for j in range(range_start[1], range_end[1] + 1)]
# [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (2, 1), (2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (3, 1), (3, 2), (3, 3), (3, 4), (3, 5), (3, 6), (4, 1), (4, 2), (4, 3), (4, 4), (4, 5), (4, 6), (5, 1), (5, 2), (5, 3), (5, 4), (5, 5), (5, 6), (6, 1), (6, 2), (6, 3), (6, 4), (6, 5), (6, 6)]
# print(grid_pinpoints)
# import pdb; pdb.set_trace()

# grid_pinpoints = [[dim * 384 for dim in pair] for pair in grid_pinpoints]


image_file = [x for x in range(16)]
step = len(image_file) / 8
sample_list = [image_file[int(i * step)] for i in range(8)]

import pdb; pdb.set_trace()