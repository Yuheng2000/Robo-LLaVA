import tqdm
import json
import copy
import os
import logging
import random
import numpy as np

import torch
import transformers
from dataclasses import dataclass, field
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import llava.benchmark.data as bmk

from llava.mm_utils import get_model_name_from_path
from llava.benchmark import data_generative
from llava.model.builder import load_pretrained_model
from llava.benchmark.eval import EvaluationArguments
from llava.train.train import update_data_args

UINFO = logging.INFO + 1

@dataclass
class EvaluationArgumentsGenerative(EvaluationArguments):
    normalize_text: bool = field(default=True,metadata={"help": "Whether text preprocessing is used in index calculation"},)
    temperature: float = field(default=0.0, metadata={"help": "List of benchmarks to evaluate."})
    top_p: float = field(default=0.95, metadata={"help": "List of benchmarks to evaluate."})
    num_beams: int = field(default=1, metadata={"help": "List of benchmarks to evaluate."})
    max_new_tokens: int = field(default=64, metadata={"help": "max_new_tokens"})

def create_model(rank, model_path):
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, _, _ = load_pretrained_model(model_path, None, model_name, device_map=rank, torch_dtype="auto")
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    return ddp_model, model, tokenizer

def run(dataloader: DataLoader, model, tokenizer, eval_args):
    rank, world_size = dist.get_rank(), dist.get_world_size()
    model.eval()
    dataset = dataloader.dataset
    pbar_desc = f"{dataset.data_args.bmk_name}@bs{dataset.data_args.batch_size}"
    pbar = tqdm.tqdm(dataloader, desc=pbar_desc, ncols=80) if rank == 0 else dataloader
    results = []

    for _, batch in enumerate(pbar):
        batch_indices = batch.pop("indices")
        input_ids = batch["input_ids"].to(device=rank)
        batch = bmk.prepare_inputs(batch, dtype=model.module.config.torch_dtype, device=rank)
        images = batch.pop("images")
        with torch.inference_mode():
            output_ids = model.module.generate(
                input_ids,
                images=images,
                do_sample=False,
                temperature=eval_args.temperature,
                top_p=eval_args.top_p,
                num_beams=eval_args.num_beams,
                max_new_tokens=eval_args.max_new_tokens,
                use_cache=True,
            )
        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        results.extend(list(zip(batch_indices.tolist(), outputs)))

    # Build the final list of results
    fnl_results = []
    for original_id, generated_text in results:
        sample = dataset.get_sample(original_id)
        fnl_results.append({
            "unique_id": original_id, # Deduplication is done with unique id
            "question_id": sample["id"], # For easy mapping of results to the original data
            "question": sample["question"],
            "pred": generated_text,
            "gt": sample["answer"].split("[SEG]"),
            "dataset_name": dataset.data_args.bmk_name,
            "type_level_1": sample.get("type_level_1", "none"),
            "type_level_2": sample.get("type_level_2", "none")
            })

    if not fnl_results:
        return None

    # Synchronize data from all devices together
    dist.barrier()
    num_samples = torch.tensor(len(fnl_results), dtype=torch.int, device=rank)
    all_rank_num_samples = [torch.tensor(0, dtype=torch.int, device=rank) for _ in range(world_size)]
    dist.all_gather(all_rank_num_samples, num_samples)
    total_samples = sum(all_rank_num_samples).item()
    all_rank_results = [None] * total_samples
    dist.all_gather_object(all_rank_results, fnl_results)

    def remove_duplicates(lst):
        seen = set()
        unique_lst = []
        for item in lst:
            key = item["unique_id"]
            if key not in seen:
                seen.add(key)
                unique_lst.append(item)
        return unique_lst

    # Process all results in the main process
    if rank == 0:
        final_results = [result for result in all_rank_results if result is not None]
        final_results = [item for sublist in final_results for item in sublist]
        final_results = remove_duplicates(final_results)
        result_json_path = os.path.join(eval_args.save_dir, f"{dataset.data_args.bmk_name}.json")
        with open(result_json_path, "w", encoding="utf-8") as out_file:
            json.dump(final_results, out_file, indent=2, ensure_ascii=False)

        return final_results
    return None

def load_benchmark(rank, bmk_name, data_args, tokenizer):
    torch.cuda.set_device(rank)
    data_args = copy.deepcopy(data_args)
    data_args.bmk_name = bmk_name
    return data_generative.build_benchmark_generative(data_args, tokenizer)

def main(rank, world_size, eval_args, data_args):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model, ori_model, tokenizer = create_model(rank, eval_args.model_dir)
    print(f"Number of images of a sample is limited to {data_args.max_num_images}.")

    data_args = update_data_args(data_args, ori_model)

    for eval_bmk in eval_args.benchmarks:
        dataloader = load_benchmark(rank, eval_bmk, data_args, tokenizer)
        run(dataloader, model, tokenizer, eval_args)

    dist.destroy_process_group()

if __name__ == "__main__":
    parser = transformers.HfArgumentParser([EvaluationArgumentsGenerative, data_generative.BenchmarkDataArguments])
    eval_args, data_args = parser.parse_args_into_dataclasses()
    if not eval_args.save_dir:
        eval_args.save_dir = os.path.join(eval_args.model_dir, 'eval', 'generative')
    if not os.path.exists(eval_args.save_dir):
        os.makedirs(eval_args.save_dir, exist_ok=True)

    WORLD_SIZE = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(eval_args.server_port)

    eval_args.benchmarks = eval(eval_args.benchmarks)
    logging.log(UINFO, f"Starting evaluation model {eval_args.model_dir} on following {len(eval_args.benchmarks)} benchmarks:\n{', '.join(eval_args.benchmarks)}")
    mp.spawn(main, args=(WORLD_SIZE, eval_args, data_args, ), nprocs=WORLD_SIZE, join=True)
