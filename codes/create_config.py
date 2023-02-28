from pathlib import Path
import argparse
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path", type=str, default="train.txt", help="Train Dataset Path"
)
parser.add_argument(
    "--valid_path", type=str, default="val.txt", help="Valid Dataset Path"
)
parser.add_argument("--run_name", type=str, required=True, help="Experiment run name")
parser.add_argument("--learning_rate", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--fp16", type=bool, default=False, help="Use mixed precision")
parser.add_argument("--use8bit", type=bool, default=True, help="Use 8-bit optimizer")
parser.add_argument(
    "--save_total_limit", type=int, default=0, help="Total number of checkpoints"
)
parser.add_argument(
    "--save_training_states", type=bool, default=False, help="Save all training states"
)

args = parser.parse_args()


def txt_file_lines(p: str) -> int:
    return len(Path(p).read_text().strip().split("\n"))


def div_spillover(n: int, bs: int) -> int:  # returns new batch size
    epoch_steps, remain = divmod(n, bs)
    if epoch_steps * 2 > bs:
        return bs  # don't bother optimising this stuff if epoch_steps are high
    if not remain:
        return bs  # unlikely but still

    if remain * 2 < bs:  # "easier" to get rid of remainder -- should increase bs
        target_bs = n // epoch_steps
    else:  # easier to increase epoch_steps by 1 -- decrease bs
        target_bs = n // (epoch_steps + 1)
    assert n % target_bs < epoch_steps + 2  # should be very few extra
    return target_bs


if __name__ == "__main__":
    DEFAULT_TRAIN_BS = 64
    DEFAULT_VAL_BS = 32

    training_samples = txt_file_lines(args.dataset_training_path)
    val_samples = txt_file_lines(args.validation_dataset_training_path)

    if training_samples < DEFAULT_TRAIN_BS:
        train_bs = training_samples
    else:
        train_bs = div_spillover(training_samples, DEFAULT_TRAIN_BS)
    if val_samples < DEFAULT_VAL_BS:
        val_bs = val_samples
    else:
        val_bs = div_spillover(val_samples, DEFAULT_VAL_BS)

    steps_per_epoch = training_samples // train_bs
    lr_decay_epochs = [20, 40, 56, 72]
    lr_decay_steps = [steps_per_epoch * e for e in lr_decay_epochs]
    print_freq = min(100, max(20, steps_per_epoch))
    val_freq = save_checkpoint_freq = print_freq * 3

    print("===CALCULATED SETTINGS===")
    print(f"train_bs={train_bs}, val_bs={val_bs}")
    print(f"lr_decay_steps={lr_decay_steps}")

    with open("../experiments/train_gpt.yml") as f:
        config = yaml.load(f)

    config["datasets"]["train"]["path"] = args.train_path
    config["datasets"]["train"]["batch_size"] = train_bs
    config["datasets"]["val"]["path"] = args.valid_path
    config["datasets"]["val"]["batch_size"] = val_bs
    config["steps"]["gpt_train"]["optimizer_params"]["lr"] = args.learning_rate
    config["train"]["val_freq"] = val_freq
    config["train"]["gen_lr_steps"] = lr_decay_steps
    config["fp16"] = args.fp16
    config["use_8bit"] = args.use8bit
    config["name"] = args.run_name
    config["logger"]["print_freq"] = print_freq
    config["logger"]["save_checkpoint_freq"] = save_checkpoint_freq
    config["logger"]["disable_state_saving"] = args.save_training_states
    config["upgrades"]["number_of_checkpoints_to_save"] = args.save_total_limit

    with open("../experiments/train_gpt.yml", "w") as f:
        yaml.dump(config, f)
