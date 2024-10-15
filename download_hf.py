from huggingface_hub import snapshot_download

# 指定模型名称和保存路径
model_name = "lmms-lab/llava-onevision-qwen2-7b-ov"
save_directory = "/home/jiyuheng/model_zoo/LLaVA-NeXT"

# 下载模型
snapshot_download(repo_id=model_name, cache_dir=save_directory)
