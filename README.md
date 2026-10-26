# DrDPO

## What is this repo?

This repo includes a reference implementation of the Dr. DPO algorithm for training language models from preference data, as described in the paper `Towards Robust Alignment of Language Models: Distributionally Robustifying Direct Preference Optimization`

The DrDPO pipeline has two stages:

1. Run supervised fine-tuning (SFT) on the dataset(s) of interest.
2. Run preference learning on the model from step 1, using preference data (ideally from the same distribution as the SFT examples).

The files in this repo are:
- `train.py`: the main entry point for training (either SFT or DrDPO preference-based training)
- `trainers.py`: the trainer classes (e.g., implementing the loop of learning as well as multi-GPU logic)
- `utils.py`: some convenience functions used by multiple other files
- `preference_datasets.py`: dataset processing logic for both SFT and DrDPO preference-based training; **this is where you'll need to make some additions to train on your own data**

## A complete example

Let's work through a complete example training pythia 2.8B on the Anthropic-HH dataset.

### Step 1: Set up environment

First, create a virtualenv and install the dependencies. Python 3.8+ is recommended.
```sh
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Step 2: Run SFT

We'll take advantage of FSDP's mixed precision in bfloat16 to speed up training; we usually see about a 50% speedup. By default, SFT will run for a single epoch over a mixture of the selected datasets. Datasets will be downloaded on the fly and cached locally.
```sh
python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_drdpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16
```
> Note: this command is run on a machine with 4 80GB A100s; on this hardware, SFT takes about 1hr 30min. If you have less compute available, you might need to increase the number of gradient accumulation steps, and SFT will take longer.

### Step 3: Run DrDPO

Check either wandb (if enabled, it is by default) or your output log to find the local run directory. To run DrDPO, you'll need the path to the final weights, which will look something like `/some/cache/dir/YOUR_USERNAME/pythia28_hh_sft_bf16_2023-06-21_16-58-17_973996/LATEST/policy.pt`. The `LATEST` directory contains the final set of weights from the end of training; periodic evaluation does not save intermediate checkpoints.
```sh
python -u train.py model=pythia28 datasets=[hh] loss=dpo loss.beta=0.1 exp_name=anthropic_drdpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt loss.mode_loss=DrDPO loss.mode_weight=1.0 
```
In the context of a label flipping setting, the only requisites are modifying the `datasets` and specifying the flipping proportion, denoted as `noise_rate`. For illustrative purposes, let's consider a `noise_rate` of 0.2 as an example:
```sh
python -u train.py model=pythia28 datasets=[hh_pairwise_noise] loss=dpo loss.beta=0.1 exp_name=anthropic_drdpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt loss.mode_loss=DrDPO loss.mode_weight=1.0 noise_rate=0.2
```

On 4 80GB A100s, DrDPO training took about 2hrs 45min.

### Run SAMDPO

SAMDPO follows `simple_DPO_SAM.pdf`: ordinary mean DPO plus half the squared norm of the virtual output-mapping gradient projected onto a detached gradient-aware basis. It does not use DrDPO reweighting or classic SAM parameter perturbations.

SAMDPO trains all Transformer parameters directly, without PEFT or LoRA. The output head and original SFT reference remain frozen. Independent input embeddings train; embeddings tied to the output head remain frozen because they share the same parameter. The virtual gradient includes only the functional output-mapping path. The policy loads and trains in FP32 regardless of `model.policy_dtype`, avoiding mixed-dtype operations and low-precision optimizer updates; `model.reference_dtype` still controls reference precision.

Single GPU (use CUDA_VISIBLE_DEVICES to select it):

```sh
CUDA_VISIBLE_DEVICES=0 python -u train.py model=pythia28 datasets=[hh] loss=samdpo exp_name=hh_samdpo trainer=BasicTrainer batch_size=4 eval_batch_size=4 gradient_accumulation_steps=1 sample_during_eval=false model.archive=/path/to/sft/LATEST/policy.pt model.policy_dtype=bfloat16 model.reference_dtype=bfloat16
```

Ordinary DDP on all visible GPUs (the entry point launches one process per GPU; do not wrap this command in torchrun):

```sh
CUDA_VISIBLE_DEVICES=0,1 python -u train.py model=pythia28 datasets=[hh] loss=samdpo exp_name=hh_samdpo_ddp trainer=DDPTrainer batch_size=8 eval_batch_size=8 gradient_accumulation_steps=1 sample_during_eval=false model.archive=/path/to/sft/LATEST/policy.pt model.policy_dtype=bfloat16 model.reference_dtype=bfloat16
```

`batch_size` and `eval_batch_size` are global pair counts and must divide evenly across ranks. Each rank independently computes its local mean DPO gradient, random sketch and basis. Only model gradients and logging metrics communicate across ranks. The optimized objective is the average of LOCAL regularized losses, not the squared norm of a global gradient. Every rank stores a complete frozen policy base and reference, so DDP memory requirements differ from FSDP.

Remove all `loss.lora.*` overrides from previous commands. Model dropout is disabled. With ordinary DDP, every rank stores the complete trainable policy, its gradients and optimizer states as well as the frozen reference; memory use is higher than LoRA. Set `loss.sam.reg_weight=0` for a matching Transformer DPO control (the current implementation still computes the penalty).

The regularizer always executes in FP32 with autocast disabled. Its hidden states, probabilities and DPO coefficients retain their graph; only basis generation uses no_grad. `loss.sam.sketch_dim` is between 1 and hidden dimension; `refresh_interval` counts optimizer steps; `orth_eps` is a relative SVD rank tolerance. At numerical rank zero the penalty is zero. Evaluation uses the existing basis without changing it; before the first training step it reports a zero penalty. Metrics distinguish train/eval DPO, SAM penalty, total loss and basis rank.

SAMDPO requires `label_smoothing=0`, `reference_free=false`, `gradient_accumulation_steps=1`, and `activation_checkpointing=false`. FSDP and TensorParallel are explicitly rejected for this loss. Existing DPO/DrDPO training remains available through their original trainers.

Checkpoint directories contain `samdpo.pt` (format version 2, full policy `state`, reference source and run signature) and `rank-N.pt` for every rank (basis, refresh step, optimizer/scheduler, counters, data progress and CPU/CUDA/Python/NumPy RNG states). Resume with the same command and add:

```sh
loss.resume=/path/to/samdpo/run/LATEST
```

Keep the same original `model.archive` SFT source to initialize the reference. `loss.resume` restores the complete trained policy and training state. The original model/SFT files must remain unchanged and accessible. Resume requires the same world size and model/data/loss/optimizer settings; the iterator is replayed to the saved batch position. Older LoRA checkpoints cannot resume this parameterization. For policy initialization only, the new `samdpo.pt` can also be used as `model.archive` because it contains the full `state`. Evaluation already completed at a saved step is not repeated.

Numerical and integration validation (no model download is required):

```sh
python -m unittest discover -s tests -v
```

The suite checks analytical GC against autograd, penalty parameter derivatives with a fixed detached basis, real PEFT LoRA with frozen tied embeddings, token shifting/masking, cancellation and zero rank, basis refresh/evaluation, mixed precision, checkpoint continuation and two-process CPU/Gloo gradient averaging. GPU NCCL training should additionally be verified on the target multi-GPU host.

## Key Implementation Details:

Owing to their straightforward structures, these loss functions can be implemented concisely within a few lines of code:

```python
# losses: batch size (loss of each pair)
# loss_config.mode_weight: $\beta^'$ in the paper
if loss_config.mode_loss == "DrDPO":
    return - loss_config.mode_weight * torch.log(torch.mean(torch.exp( - losses / loss_config.mode_weight))), metrics
else:
    return losses.mean(), metrics
```

## Acknowledgement
The project is built upon [DPO](https://github.com/eric-mitchell/direct-preference-optimization)


For a local HH dataset, run from the repository root with `python train.py`,
not an imported `train.main()` inside `python -c`. Set
`dataset_path=/mnt/LLM/datasets/hh-rlhf` and
`model.name_or_path=/mnt/LLM/models/pythia-2.8b`
`model.tokenizer_name_or_path=/mnt/LLM/models/pythia-2.8b`.
Local HH supports a saved DatasetDict, saved train/test subdirectories, or raw
train.json[.gz]/test.json[.gz] files (including nested category directories).
DDPTrainer spawns its own workers; select cards with
`CUDA_VISIBLE_DEVICES=4,5,6,7` and do not wrap this launch in torchrun.


### Local four-GPU SFT before SAMDPO

On the Linux server, run `bash train_sft_4x4090.sh` from this checkout.
The script selects GPUs 4/5/6/7, reads the local Pythia and HH directories,
and uses full-parameter FSDP with FP32 master weights, BF16 mixed precision,
activation checkpointing and RMSprop. A global batch of 16 accumulated over
4 microbatches gives one sequence per GPU per microbatch. No torchrun is needed.
The script captures stdout/stderr in `logs/*.log`; training outputs are under
`.cache/<username>/hh_sft_pythia28_4x4090_<timestamp>/`.

`metrics.jsonl` records EVERY optimizer update and each evaluation, even when
W&B is disabled or console logging is throttled. SFT logs sequence-summed loss,
response-token-weighted NLL/perplexity, learning rate, gradient norm and speed.
Evaluation (without intermediate checkpoint saves) runs initially, every 2000 training EXAMPLES (125 updates with this
batch), and after the final SFT update. Test NLL/perplexity assess next-token
prediction, not human preference or conversational quality.

`eval_every` controls evaluation only for SFT: no intermediate model checkpoints
are written. After training, `LATEST/policy.pt` is saved once and contains final
model weights and final SFT evaluation metrics. Optimizer/scheduler states are
not saved for SFT. FP32 Pythia-2.8B weights occupy approximately 11 GB.
Training logs and evaluation history still persist throughout the run.
SFT does not implement training resume; exact resume is currently SAMDPO only.

The default script disables generation for faster FSDP evaluation. Override
with `bash train_sft_4x4090.sh sample_during_eval=true n_eval_model_samples=4`
to save prompt/response samples in `samples/step-N.json`; this is substantially
slower with FSDP. No best-checkpoint selection is performed automatically.
To start SAMDPO, use the final SFT `LATEST/policy.pt`
as `model.archive`, keeping the original model directory as `model.name_or_path`.


### Evaluation and final-only checkpoints (all losses)

SFT, DPO/DrDPO, IPO and SAMDPO now evaluate periodically without saving any
intermediate checkpoints. After the final update, evaluation runs again and
`worker_main` saves exactly once under `LATEST/`, including final metrics.
SFT retains only `policy.pt`; DPO/DrDPO retain policy/optimizer/scheduler states;
SAMDPO retains the full policy and per-rank training/subspace states.
There is no automatic best-checkpoint selection or recovery from an interrupted
run before the final save. Existing checkpoint files are not deleted.

HH uses its supplied `train` and `test` splits (including local files); no new
random split or separate validation set is created. By default, evaluation uses
a deterministic subset of 1000 unique test pairs. Set `n_eval_examples=null` for one
pass over test data; evaluation never repeats epochs to fill the requested size.
Only complete global eval batches are used, so a trailing incomplete batch is
dropped. `eval/examples` records the actual evaluated count. Repeated monitoring
of this test subset is not a separate held-out final test evaluation.

For DPO/DrDPO/SAMDPO, `preference_acc/eval` is an explicit alias of
`rewards_eval/accuracies`: the fraction of pairs with
`beta * (log pi(chosen) - log ref(chosen)) >
 beta * (log pi(rejected) - log ref(rejected))`.
`policy_preference_acc/eval` instead compares the two policy sequence log
probabilities directly. Both use summed response-token log probabilities,
exclude prompt/padding via the label mask, and count ties as incorrect.
Per-pair values are gathered across ranks before averaging over batches,
so metrics are weighted by evaluated pair count. SFT only computes chosen-response
NLL/perplexity, not preference accuracy. These are offline pair-ranking metrics,
not generated-answer win rates. When policy equals reference, implicit rewards
tie and `preference_acc/eval` is zero by this convention.


HH evaluation defaults to 1000 unique test pairs. All test prompts are reserved:
training rows with the same prompt (Unicode NFKC + whitespace normalization) are
excluded, including HH pairwise-noise and SFT input paths. Duplicate test pairs,
including reversed chosen/rejected order, are counted once. An undersized test
split raises an error rather than repeating examples or borrowing train data.
The audit checks all HH train/test rows before tokenization and writes raw/clean
counts to `dataset_audit.json`; it cannot certify older SFT/model training data
or rule out semantic paraphrases or overlap created by token truncation.

Check a server dataset before training:
`python hh_split_audit.py --dataset-path /mnt1/zhouyeying/datasets/hh-rlhf`.
The local code checkout alone does not contain the server dataset, so its exact
counts must be read from this command or the startup audit. Evaluation seeds are
identical across losses. Set `n_eval_examples=1000 eval_batch_size=4` on four GPUs:
250 global eval batches, one pair per GPU per batch, 1000 global pairs total.
Per-pair metrics use all_gather before averaging, never an unweighted average
of per-rank accuracies. Only rank zero writes the final global metrics log.
If an existing shell script overrides `n_eval_examples=256`, update that override.
`n_eval_examples` must divide evenly into `eval_batch_size` (or be null).
Repeated evaluation uses the same fixed subset and performs no parameter updates.
# DPO
