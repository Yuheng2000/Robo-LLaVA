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
export WANDB_MODE=disabled #不同步到wandb，没外网
############### Pretrain ################

PROMPT_VERSION="qwen_1_5"
BASE_RUN_NAME="llava-onevision-qwen2-7b-ov"
#"llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"

CKPT_PATH=$LLM_VERSION # this could also be the previous stage checkpoint

BASE_MODEL_PATH='/share/henry/LLaVA-NeXT/checkpoints/llava-onevision-qwen2-7b-ov'

NUM_GPUS=8
NNODES=1
MID_RUN_NAME='llava-onevision-qwen2-7b-ov-robovqa-v3'
HOSTNAME=$(hostname)

deepspeed --num_gpus $NUM_GPUS --num_nodes $NNODES \
    --hostfile hostfile \
    --master_addr 10.9.200.93 \
    --master_port 65431 \
    --ssh_port 2288 llava/train/train_mem.py \
    --model_name_or_path ${BASE_MODEL_PATH} \
    --version ${PROMPT_VERSION} \
    --data_path ./ego-robo.yaml \
    --image_folder ./onevision_data/images \
    --video_folder ./ego-robo_data/videos/robovqa \
    --mm_tunable_parts="mm_vision_tower,mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --group_by_modality_length True \
    --image_aspect_ratio anyres_max_9 \
    --image_grid_pinpoints  "\"(1x1),...,(6x6)\"" \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name $MID_RUN_NAME \
    --output_dir "./checkpoints/${MID_RUN_NAME}" \
    --num_train_epochs 2 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --frames_upbound 32  \
    --deepspeed scripts/zero2.json \
    >deepspeed_1.log 2>&1

# You can delete the sdpa attn_implementation if you want to use flash attn
