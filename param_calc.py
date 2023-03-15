# stolen from the colab, will probably go out of sync
from pathlib import Path
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('--train', type=Path, required=True)
parser.add_argument('--valid', type=Path, required=True)
parser.add_argument('--bs_train', type=int, default=64)
parser.add_argument('--bs_val', type=int, default=16)
parser.add_argument('--first_decay', type=int, default=20)
args = parser.parse_args()

from pathlib import Path
DEFAULT_TRAIN_BS = args.bs_train
DEFAULT_VAL_BS = args.bs_val
FIRST_DECAY = args.first_decay
#@markdown # Hyperparameter calculation
#@markdown Run this cell to obtain suggested parameters for training
Dataset_Training_Path = args.train #@param {type:"string"}
ValidationDataset_Training_Path = args.valid #@param {type:"string"}

#@markdown ### **NOTE**: Dataset must be in the following format.

#@markdown  `dataset/`
#@markdown * ---├── `val.txt`
#@markdown * ---├── `train.txt`
#@markdown * ---├── `wavs/`

#@markdown `wavs/` directory must contain `.wav` files.

#@markdown  Example for `train.txt` and `val.txt`:

#@markdown * `wavs/A.wav|Write the transcribed audio here.`

#@markdown todo: actually check the dataset structure

if Dataset_Training_Path == ValidationDataset_Training_Path:
  print("WARNING: training dataset path == validation dataset path!!!")
  print("\tThis is technically okay but will make all of the validation metrics useless. ")

def txt_file_lines(p: str) -> int:
  return len(Path(p).read_text().strip().split('\n'))
training_samples = txt_file_lines(Dataset_Training_Path)
val_samples = txt_file_lines(ValidationDataset_Training_Path)

if training_samples < 128: print("WARNING: very small dataset! the smallest dataset tested thus far had ~200 samples.")
if val_samples < 20: print("WARNING: very small validation dataset! val batch size will be scaled down to account")

def div_spillover(n: int, bs: int) -> int: # returns new batch size
  epoch_steps,remain = divmod(n,bs)
  if epoch_steps*2 > bs: return bs # don't bother optimising this stuff if epoch_steps are high
  if not remain: return bs # unlikely but still

  if remain*2 < bs: # "easier" to get rid of remainder -- should increase bs
    target_bs = n//epoch_steps
  else: # easier to increase epoch_steps by 1 -- decrease bs
    target_bs = n//(epoch_steps+1)
  assert n%target_bs < epoch_steps+2 # should be very few extra 
  return target_bs

if training_samples < DEFAULT_TRAIN_BS:
  print("WARNING: dataset is smaller than a single batch. This will almost certainly perform poorly. Trying anyway")
  train_bs = training_samples
else:
  train_bs = div_spillover(training_samples, DEFAULT_TRAIN_BS)
if val_samples < DEFAULT_VAL_BS:
  val_bs = val_samples
else:
  val_bs = div_spillover(val_samples, DEFAULT_VAL_BS)

steps_per_epoch = training_samples//train_bs
lr_decay_epochs = [FIRST_DECAY, FIRST_DECAY*2, FIRST_DECAY*14//5, FIRST_DECAY*18//5]
lr_decay_steps = [steps_per_epoch * e for e in lr_decay_epochs]
print_freq = min(100, max(20, steps_per_epoch))
val_freq = save_checkpoint_freq = print_freq * 3

print("===CALCULATED SETTINGS===")
print(f'{train_bs=} {val_bs=}')
print(f'{val_freq=} {lr_decay_steps=}')
print(f'{print_freq=} {save_checkpoint_freq=}')
