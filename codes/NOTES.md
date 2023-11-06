# notes
many files in DLAS will run if you load them as modules, e.g.

```python
$ py -m data.audio.paired_voice_audio_dataset
```

## dataloaders

./data/audio/:
- audio_with_noise_dataset.py, unsupervised_audio_dataset.py: -- seem interesting for unlabelled audio data
- gpt_tts_dataset.py: -- loads quantized mels instead of wavs. useful if you preprocess your dataset with the vqvae first.
- grand_conjoined_dataset.py -- this is the grand dataloader neonbjb used. Not useful for most people because it expects ocotillo files
- nv_tacotron_dataset.py -- related to ljspeech and commonvoice and voxpopuli datasets
- paired_voice_audio_dataset.py -- For ordinary TTS datasets; use this!

./scripts/audio/preparation/pipeline.py: seems useful for parsing files

## "Injectors"
Injectors manipulate the input dataset on-the-fly. They're defined in ./trainer/injectors/* && ./trainer/inject.py. Note this function:
```python
def format_injector_name(name):
    # Formats by converting from CamelCase to snake_case and removing trailing "_injector"
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    name = re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    return name.replace("_injector", "")
```
Means an injector like `TorchMelSpectrogramInjector` (defined in `./trainer/injectors/audio_injectors.py`) will be converted to `torch_mel_spectrogram` in the .yml file:

```yml
    injectors:    
      paired_to_mel:
        type: torch_mel_spectrogram
        mel_norm_file: ../experiments/clips_mel_norms.pth
        in: wav
        out: paired_mel
      paired_cond_to_mel:
        type: for_each
        subtype: torch_mel_spectrogram
        mel_norm_file: ../experiments/clips_mel_norms.pth
        in: conditioning
        out: paired_conditioning_mel
      to_codes:
        type: discrete_token
        in: paired_mel
        out: paired_mel_codes
      paired_fwd_text:
        type: generator
        generator: gpt
        in: [paired_conditioning_mel, padded_text, text_lengths, paired_mel_codes, wav_lengths]
        out: [loss_text_ce, loss_mel_ce, logits]      
```



## EXAMPLE yml file
put this file in `DL-Art-School/experiments/gpt.yml`

```yml
name: CHANGEME_your_experiment_name
model: extensibletrainer
scale: 1
gpu_ids: [0] # <-- unless you have multiple gpus, use this
start_step: -1
checkpointing_enabled: true  # <-- Gradient checkpointing. Enable for huge GPU memory savings. Disable for distributed training.
fp16: false # might want to check this out
wandb: false  # <-- enable to log to wandb. tensorboard logging is always enabled.
use_tb_logger: true

datasets:
  train:
    name: CHANGEME_training_dataset_name
    n_workers: 8 # idk what this does
    batch_size: 128 # This leads to ~16GB of vram usage on my 3090.
    mode: paired_voice_audio
    path: CHANGEME_path_to_training_dataset
    fetcher_mode: ['lj'] # CHANGEME if your dataset isn't in LJSpeech format
    phase: train
    max_wav_length: 255995
    max_text_length: 200
    sample_rate: 22050
    load_conditioning: True
    num_conditioning_candidates: 2
    conditioning_length: 44000
    use_bpe_tokenizer: True
    load_aligned_codes: False
  val:
    name: CHANGEME_validation_dataset_name
    n_workers: 1
    batch_size: 32 # this could be higher probably
    mode: paired_voice_audio
    path: CHANGEME_path_to_validation_dataset
    fetcher_mode: ['lj']
    phase: val # might be broken idk
    max_wav_length: 255995
    max_text_length: 200
    sample_rate: 22050
    load_conditioning: True
    num_conditioning_candidates: 2
    conditioning_length: 44000
    use_bpe_tokenizer: True
    load_aligned_codes: False

steps:        
  gpt_train:
    training: gpt
    loss_log_buffer: 500 # no idea what this does

    # Generally follows the recipe from the DALLE paper.
    optimizer: adamw # this should be adamw_zero if you're using distributed training
    optimizer_params:
      lr: !!float 1e-5 # CHANGEME: this was originally 1e-4; I reduced it to 1e-5 because it's fine-tuning, but **you should experiment with this value**
      weight_decay: !!float 1e-2
      beta1: 0.9
      beta2: 0.96
    clip_grad_eps: 4

    injectors:  # TODO: replace this entire sequence with the GptVoiceLatentInjector
      paired_to_mel:
        type: torch_mel_spectrogram
        mel_norm_file: ../experiments/clips_mel_norms.pth
        in: wav
        out: paired_mel
      paired_cond_to_mel:
        type: for_each
        subtype: torch_mel_spectrogram
        mel_norm_file: ../experiments/clips_mel_norms.pth
        in: conditioning
        out: paired_conditioning_mel
      to_codes:
        type: discrete_token
        in: paired_mel
        out: paired_mel_codes
        dvae_config: "../experiments/train_diffusion_vocoder_22k_level.yml" # EXTREMELY IMPORTANT
      paired_fwd_text:
        type: generator
        generator: gpt
        in: [paired_conditioning_mel, padded_text, text_lengths, paired_mel_codes, wav_lengths]
        out: [loss_text_ce, loss_mel_ce, logits]      
    losses:
      text_ce:
        type: direct
        weight: .01
        key: loss_text_ce
      mel_ce:
        type: direct
        weight: 1
        key: loss_mel_ce

networks:
  gpt:
    type: generator 
    which_model_G: unified_voice2 # none of the unified_voice*.py files actually match the tortoise inference code... 4 and 3 have "alignment_head" (wtf is that?), 2 lacks the types=1 parameter.
    kwargs:
      layers: 30 # WAS 8
      model_dim: 1024 # WAS 512
      heads: 16 # WAS 8
      max_text_tokens: 402 # WAS 120
      max_mel_tokens: 604 # WAS 250
      max_conditioning_inputs: 2 # WAS 1
      mel_length_compression: 1024
      number_text_tokens: 256 # supposed to be 255 for newer unified_voice files 
      number_mel_codes: 8194
      start_mel_token: 8192
      stop_mel_token: 8193
      start_text_token: 255
      train_solo_embeddings: False # missing in uv3/4
      use_mel_codes_as_input: True # ditto
      checkpointing: True
      #types: 1 # this is MISSING, but in my analysis 1 is equivalent to not having it.
      #only_alignment_head: False  # uv3/4

path:
  pretrain_model_gpt: '../experiments/autoregressive.pth' # CHANGEME: copy this from tortoise cache
  strict_load: true
  #resume_state: ../experiments/train_imgnet_vqvae_stage1/training_state/0.state   # <-- Set this to resume from a previous training state.

# afaik all units here are measured in **steps** (i.e. one batch of batch_size is 1 unit)
train: # CHANGEME: ALL OF THESE PARAMETERS SHOULD BE EXPERIMENTED WITH
  niter: 50000
  warmup_iter: -1
  mega_batch_factor: 4    # <-- Gradient accumulation factor. If you are running OOM, increase this to [2,4,8].
  val_freq: 500

  default_lr_scheme: MultiStepLR
  gen_lr_steps: [500, 1000, 1400, 1800] #[50000, 100000, 140000, 180000]
  lr_gamma: 0.5

eval:
  output_state: gen
  injectors:
    gen_inj_eval:
      type: generator
      generator: generator
      in: hq
      out: [gen, codebook_commitment_loss]

logger: 
  print_freq: 100
  save_checkpoint_freq: 500 # CHANGEME: especially you should increase this it's really slow
  visuals: [gen, mel]
  visual_debug_rate: 500
  is_mel_spectrogram: true
```

then you can run 
```python
$ py train.py -opt ../experiments/gpt.yml 
```
to train
