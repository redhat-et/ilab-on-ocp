from instructlab.training import (
    run_training,
    TorchrunArgs,
    TrainingArgs,
    DeepSpeedOptions,
)

# define training-specific arguments
training_args = TrainingArgs(
    deepspeed_options = DeepSpeedOptions(cpu_offload_optimizer=True),
    # define data-specific arguments
    model_path = "ibm-granite/granite-7b-base",
    data_path = "training/sample-data/train_all_pruned_SDG.jsonl",
    ckpt_output_dir = "data/saved_checkpoints",
    data_output_dir = "data/outputs",

    # define model-trianing parameters
    max_seq_len = 4096,
    max_batch_len = 20000,
    num_epochs = 2,
    effective_batch_size = 3840,
    save_samples = 0,
    learning_rate = 2e-6,
    warmup_steps = 800,
    is_padding_free = True, # set this to true when using Granite-based models
    random_seed = 42,
    checkpoint_at_epoch = True,
)

torchrun_args = TorchrunArgs(
    nnodes = 1, # number of machines 
    nproc_per_node = 2, # num GPUs per machine
    node_rank = 0, # node rank for this machine
    rdzv_id = 123,
    rdzv_endpoint = '127.0.0.1:12345',
)

run_training(
    torch_args=torchrun_args,
    train_args=training_args,
)