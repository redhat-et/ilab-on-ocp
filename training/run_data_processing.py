import argparse
import os

from instructlab.training import (
    TrainingArgs,
    DataProcessArgs,
)
import instructlab.training.data_process as dp

def data_processing(train_args: TrainingArgs) -> None:
    # early validation logic here
    if train_args.max_batch_len < train_args.max_seq_len:
        raise ValueError(
            f"the `max_batch_len` cannot be less than `max_seq_len`: {train_args.max_batch_len=} < {train_args.max_seq_len=}"
        )
    
        # process the training data
    if not os.path.exists(train_args.data_output_dir):
        os.makedirs(train_args.data_output_dir, exist_ok=True)
    dp.main(
        DataProcessArgs(
            # XXX(osilkin): make a decision here, either:
            #   1. the CLI is fully responsible for managing where the data is written
            #   2. we never cache it and simply write it to a tmp file every time.
            #
            # An important reason for why #1 would be preferable is in the case of OpenShift/SELinux
            # where the user has a defined place for new temporary data to be written.
            data_output_path=train_args.data_output_dir,
            model_path=train_args.model_path,
            data_path=train_args.data_path,
            max_seq_len=train_args.max_seq_len,
            chat_tmpl_path=train_args.chat_tmpl_path,
        )
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Training Args
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


    args = parser.parse_args()

    # define training-specific arguments
    training_args = TrainingArgs(
        # define data-specific arguments
        model_path = args.model_path,
        data_path = args.data_path,
        ckpt_output_dir = args.ckpt_output_dir,
        data_output_dir = args.data_output_dir,

        # define model-trianing parameters
        max_seq_len = args.max_seq_len,
        max_batch_len = args.max_batch_len,

       # XXX(shanand): We don't need the following arguments 
       # for data processing. Added them for now to avoid
       # Pydantic validation errors for TrainingArgs
        num_epochs = args.num_epochs,
        effective_batch_size = args.effective_batch_size,
        save_samples = args.save_samples,
        learning_rate = args.learning_rate,
        warmup_steps = args.warmup_steps,
        is_padding_free = args.is_padding_free, # set this to true when using Granite-based models

    )

    data_processing(training_args)