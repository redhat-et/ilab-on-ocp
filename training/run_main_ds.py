import argparse
import os
import subprocess

from instructlab.training import (
    TorchrunArgs,
    TrainingArgs,
    DeepSpeedOptions,
    config
)

from instructlab.training.utils import StreamablePopen


def run_main_ds(torch_args: TorchrunArgs, train_args: TrainingArgs) -> None:
    """
    Wrapper around the main training job that calls torchrun.
    """

    if not os.path.exists(train_args.ckpt_output_dir):
        os.makedirs(train_args.ckpt_output_dir, exist_ok=True)
    command = [
        "torchrun",
        f"--nnodes={torch_args.nnodes}",
        f"--node_rank={torch_args.node_rank}",
        f"--nproc_per_node={torch_args.nproc_per_node}",
        f"--rdzv_id={torch_args.rdzv_id}",
        f"--rdzv_endpoint={torch_args.rdzv_endpoint}",
        f"-m",
        f"instructlab.training.main_ds",
        f"--model_name_or_path={train_args.model_path}",
        f"--data_path={train_args.data_output_dir}/data.jsonl",
        f"--output_dir={train_args.ckpt_output_dir}",
        f"--num_epochs={train_args.num_epochs}",
        f"--effective_batch_size={train_args.effective_batch_size}",
        f"--learning_rate={train_args.learning_rate}",
        f"--num_warmup_steps={train_args.warmup_steps}",
        f"--save_samples={train_args.save_samples}",
        f"--log_level=INFO",
        f"--max_batch_len={train_args.max_batch_len}",
        f"--seed={train_args.random_seed}",
        f"--chat-tmpl-path={train_args.chat_tmpl_path}",
    ]

    if train_args.checkpoint_at_epoch:
        command.append("--checkpoint_at_epoch")

    if train_args.mock_data:
        command.append("--mock_data")
        if train_args.mock_len:
            command.append(f"--mock_len={train_args.mock_len}")

    if train_args.is_padding_free:
        command.append("--is_granite")

    if train_args.disable_flash_attn:
        if train_args.is_padding_free:
            raise RuntimeError(
                "ERROR: Trying to use padding-free transformer without flash attention is not supported"
            )
        command.append("--disable_flash_attn")

    if train_args.lora:
        command.extend(
            [
                f"--lora_r={train_args.lora.rank}",
                f"--lora_alpha={train_args.lora.alpha}",
                f"--lora_dropout={train_args.lora.dropout}",
                "--lora_target_modules",
            ]
        )
        command.extend(train_args.lora.target_modules)
        # hard-code 4-bit quantization for now, change this when we add more
        quant_dtype = train_args.lora.quantize_data_type
        quantization_is_enabled = quant_dtype in (
            config.QuantizeDataType.NF4,
            config.QuantizeDataType.NF4.value,
        )
        if quantization_is_enabled:
            command.append("--lora_quant_bits=4")

    # deepspeed opts
    if train_args.deepspeed_options.save_samples:
        command.append(f"--save_samples_ds={train_args.deepspeed_options.save_samples}")
    if train_args.deepspeed_options.cpu_offload_optimizer:
        command.extend(
            [
                "--cpu_offload_optimizer",
                f"--cpu_offload_optimizer_ratio={train_args.deepspeed_options.cpu_offload_optimizer_ratio}",
            ]
        )
        if train_args.deepspeed_options.cpu_offload_optimizer_pin_memory:
            command.append("--cpu_offload_optimizer_pin_memory")

    print(f"\033[92mRunning command: {' '.join(command)}\033[0m")
    process = None
    try:
        process = StreamablePopen(
            f"{train_args.ckpt_output_dir}/full_logs_global{torch_args.node_rank}.log",
            command,
        )

    except KeyboardInterrupt:
        print("Process interrupted by user")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        if "process" not in locals() or process is None:
            return
        if process.poll() == 0:
            print("\033[92mOperation completed successfully! 🎉\033[0m")
        else:
            print("\033[91mOperation failed, terminating process.\033[0m")

        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            print("\033[91mProcess did not terminate in time, killing it.\033[0m")
            process.kill()


if __name__ == "__main__":
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
    parser.add_argument("--nnodes", default=os.getenv("NNODES", "1"))
    parser.add_argument("--nproc_per_node", default=os.getenv("NPROC_PER_NODE", "1"))
    parser.add_argument("--node_rank", default=os.getenv("RANK", "0"))
    parser.add_argument("--rdzv_id", default=123)
    parser.add_argument("--rdzv_endpoint", default=f"{os.getenv('MASTER_ADDR', '127.0.0.1')}:{os.getenv('MASTER_PORT','12345')}")

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

    run_main_ds(torch_args=torchrun_args, train_args=training_args)

