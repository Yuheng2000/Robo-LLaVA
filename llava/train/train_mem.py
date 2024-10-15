# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import logging
import warnings

os.environ["NCCL_DEBUG"] = "ERROR" #filter nccl info
logging.basicConfig(level=logging.ERROR)
# logger = logging.getLogger("deepspeed")  
# logger.setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("torch").setLevel(logging.ERROR)
logging.getLogger("accelerate").setLevel(logging.ERROR)

# 忽略所有 UserWarning 警告
warnings.filterwarnings("ignore", category=UserWarning)
# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)


from llava.train.train import train


os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_PROJECT"] = "llava-ov"

if __name__ == "__main__":
    train()