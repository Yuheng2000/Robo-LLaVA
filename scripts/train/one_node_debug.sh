# export OMP_NUM_THREADS=8
# export NCCL_IB_DISABLE=0
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG=INFO

LLM_VERSION="Qwen/Qwen2-7B-Instruct" 
# for 7b model we recommend bs=1, accum=2, 16 nodes, 128 gpus, lr=1e-5, warmup=0.03
# for 72b model we recommend bs=1, accum=1, 32 nodes, 256 gpus, lr=1e-5, warmup=0.03
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="qwen_1_5"

BASE_RUN_NAME="llava-onevision-qwen2-7b-ov"
#"llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint

# BASE_MODEL_PATH='/home/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov'
BASE_MODEL_PATH='/home/model_zoo/llava-onevision-qwen2-7b-ov'

NUM_GPUS=1
NNODES=1
RANK=0
# ADDR=10.9.200.87
ADDR=10.9.200.92
PORT=9204

MID_RUN_NAME='llava-onevision-qwen2-7b-ov-robovqa-v3'

#    --pretrain_mm_mlp_adapter="/share/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov/mm_projector.bin" \
# CUDA_VISIBLE_DEVICES=0 torchrun llava/train/train_mem.py --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \

    # --image_folder ../../lmz/data/llava-v1.5-instruct \
    # --video_folder ../../world_model/robovqa/videos \
    # --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    # --mm_newline_position grid \
    # --mm_spatial_pool_stride 2 \
    # --add_faster_video false \

# CUDA_VISIBLE_DEVICES=0 torchrun llava/train/train_mem.py \
deepspeed --num_gpus 8 --num_nodes 1 llava/train/train_mem.py \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ./data_config/instruct_FT.yaml \
    --image_folder /home/dataset_zoo/M4-Instruct/ \
    --video_folder /home/dataset_zoo/M4-Instruct-Videos/ \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints "\"(1x1),...,(6x6)\"" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "/home/jiyuheng/Robo-LLaVA/checkpoints/test/${MID_RUN_NAME}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4700 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --deepspeed scripts/zero2.json
    
    # deepspeed llava/train/train_mem.py \
    # --model_name_or_path ${BASE_MODEL_PATH} \
    # --version ${PROMPT_VERSION} \
    # --data_path ./data_config/instruct_FT.yaml \
    # --image_folder /home/dataset_zoo/M4-Instruct/ \
    # --video_folder /home/dataset_zoo/M4-Instruct-Videos/ \
    # --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    # --mm_vision_tower_lr=2e-6 \
    # --vision_tower ${VISION_MODEL_VERSION} \
    # --mm_projector_type mlp2x_gelu \
    # --mm_vision_select_layer -2 \
    # --mm_use_im_start_end False \
    # --mm_use_im_patch_token False \
    # --group_by_modality_length True \
    # --image_aspect_ratio anyres \
    # --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    # --mm_patch_merge_type spatial_unpad \
    # --mm_newline_position grid \
    # --mm_spatial_pool_stride 2 \
    # --add_faster_video false \
    # --bf16 True \
    # --run_name $MID_RUN_NAME \
    # --output_dir "/home/jiyuheng/Robo-LLaVA/checkpoints/test/${MID_RUN_NAME}" \
    # --num_train_epochs 2 \
    # --per_device_train_batch_size 4 \
    # --per_device_eval_batch_size 4 \
    # --gradient_accumulation_steps 4 \
    # --evaluation_strategy "no" \
    # --save_strategy "epoch" \
    # --save_total_limit 3 \
    # --learning_rate 1e-5 \
    # --weight_decay 0. \
    # --warmup_ratio 0.03 \
    # --lr_scheduler_type "cosine" \
    # --logging_steps 1 \
    # --tf32 True \
    # --model_max_length 5120 \
    # --gradient_checkpointing True \
    # --dataloader_num_workers 4 \
    # --lazy_preprocess True \
    # --report_to wandb \
    # --torch_compile True \
    # --torch_compile_backend "inductor" \
    # --dataloader_drop_last True \
    # --frames_upbound 32 \
    # --deepspeed scripts/zero2.json



# You can delete the sdpa attn_implementation if you want to use flash attn

# VISION_MODEL_VERSION="/mnt/hpfs/baaiei/hub/models--google--siglip-so400m-patch14-384"
# VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

# ############### Pretrain ################

# PROMPT_VERSION="qwen_1_5"

# BASE_RUN_NAME="llava-onevision-qwen2-7b-ov"
# #"llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
# echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

# CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint