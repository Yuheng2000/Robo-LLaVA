LLM_VERSION="Qwen/Qwen2-7B-Instruct"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="google/siglip-so400m-patch14-384"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################
#没外网，禁止wandb，否则要配置clash的yaml的rules
export WANDB_MODE=disabled

#32768  max length
PROMPT_VERSION="qwen_1_5"  #or "qwen_2"

BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"

NUM_GPUS=8
NNODES=1
LR=1e-5
BS=2
ACC_STEP=2

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")

MID_RUN_NAME=llavaov-fromScratch-2*2-visualfrozen-$NNODES-$CURRENT_TIME
echo "MID_RUN_NAME: ${MID_RUN_NAME}"

# 生成日志文件名，包含运行时间信息
LOG_FILE="runs/${MID_RUN_NAME}_bs${BS}_${LR}.log"
# 提示运行日志的位置
echo "日志文件: $LOG_FILE"


# LLM_PATH='/home/henry/mllm_checkpoints/Qwen2-7B-Instruct'
#ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${NUM_GPUS}" --nnodes="${NNODES}" --node_rank="${RANK}" --master_addr="${ADDR}" --master_port="${PORT}" \
    #CUDA_VISIBLE_DEVICES=0 
    deepspeed --num_gpus $NUM_GPUS --num_nodes $NNODES --hostfile hostfile_$NNODES llava/train/train_mem.py \
    --model_name_or_path $LLM_VERSION  \
    --version ${PROMPT_VERSION} \
    --data_path ./data_config/instruct_FT.yaml\
    --image_folder ../../lmz/data/llava-v1.5-instruct \
    --video_folder ../../world_model/robovqa/videos \
    --pretrain_mm_mlp_adapter="checkpoints/projectors/llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain/mm_projector.bin" \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower $VISION_MODEL_VERSION \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres \
    --image_grid_pinpoints "[(384, 768), (768, 384), (768, 768), (1152, 384), (384, 1152)]" \
    --mm_patch_merge_type spatial_unpad \
    --mm_newline_position grid \
    --mm_spatial_pool_stride 2 \
    --add_faster_video false \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $ACC_STEP \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate $LR \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32 \
    --deepspeed scripts/zero2.json \
    >$LOG_FILE 2>&1

# You can delete the sdpa attn_implementation if you want to use flash attn

