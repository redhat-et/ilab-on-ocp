import argparse
import os
from instructlab.training import (
    run_training,
    TorchrunArgs,
    TrainingArgs,
    DeepSpeedOptions,
)

parser = argparse.ArgumentParser()

# Training Args
parser.add_argument("--cpu_offload", default=True)
parser.add_argument("--model_path", default="ibm-granite/granite-7b-base")
parser.add_argument("--data_path", default="training/sample-data/train_all_pruned_SDG.jsonl")
parser.add_argument("--ckpt_output_dir",default="data/saved_checkpoints")
parser.add_argument("--data_output_dir", default="data/outputs")
parser.add_argument("--max_seq_len", default=4096)
parser.add_argument("--max_batch_len", default=20000)
parser.add_argument("--num_epochs", default=2)
parser.add_argument("--effective_batch_size", default=3840)
parser.add_argument("--save_samples", default=0)
parser.add_argument("--learning_rate", default=2e-6)
parser.add_argument("--warmup_steps", default=800)
parser.add_argument("--is_padding_free", default=True)
parser.add_argument("--random_seed", default=42)
parser.add_argument("--checkpoint_at_epoch", default=True)

# Torchrun Args
parser.add_argument("--nnodes", default=1)
parser.add_argument("--nproc_per_node", default=1)
parser.add_argument("--node_rank", default=0)
parser.add_argument("--rdzv_id", default=123)
parser.add_argument("--rdzv_endpoint", default= "127.0.0.1:12345")

args = parser.parse_args()

# define training-specific arguments
training_args = TrainingArgs(
    deepspeed_options = DeepSpeedOptions(cpu_offload_optimizer=args.cpu_offload),
    # define data-specific arguments
    model_path = args.model_path,
    data_path = args.data_path,
    ckpt_output_dir = args.ckpt_output_dir,
    data_output_dir = args.data_output_dir,

    # define model-trianing parameters
    max_seq_len = args.max_seq_len,
    max_batch_len = args.max_batch_len,
    num_epochs = args.num_epochs,
    effective_batch_size = args.effective_batch_size,
    save_samples = args.save_samples,
    learning_rate = args.learning_rate,
    warmup_steps = args.warmup_steps,
    is_padding_free = args.is_padding_free, # set this to true when using Granite-based models
    random_seed = args.random_seed,
    checkpoint_at_epoch = args.checkpoint_at_epoch
)

torchrun_args = TorchrunArgs(
    nnodes = args.nnodes, # number of machines 
    nproc_per_node = args.nproc_per_node, # num GPUs per machine
    node_rank = args.node_rank, # node rank for this machine
    rdzv_id = args.rdzv_id,
    rdzv_endpoint = args.rdzv_endpoint,
)

run_training(
    torch_args=torchrun_args,
    train_args=training_args,
)