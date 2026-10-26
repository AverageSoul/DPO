import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import tensor_parallel as tp
import contextlib

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir,
)
import numpy as np
import wandb
import tqdm

import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple
from samdpo import OutputGradientRegularizer, samdpo_penalty, validate_samdpo


def preference_loss(policy_chosen_logps: torch.FloatTensor,
                    policy_rejected_logps: torch.FloatTensor,
                    reference_chosen_logps: torch.FloatTensor,
                    reference_rejected_logps: torch.FloatTensor,
                    beta: float,
                    label_smoothing: float = 0.0,
                    ipo: bool = False,
                    reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.

    Args:
        policy_chosen_logps: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_logps: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_logps: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_logps: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        label_smoothing: conservativeness for DPO loss, which assumes that preferences are noisy (flipped with probability label_smoothing)
        ipo: If True, use the IPO loss instead of the DPO loss.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = reference_chosen_logps - reference_rejected_logps

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios  # also known as h_{\pi_\theta}^{y_w,y_l}

    if ipo:
        losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
    else:
        # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps).detach()
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps).detach()

    return losses, chosen_rewards, rejected_rewards


def _get_batch_logps(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = (labels != -100)

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0

    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    if average_log_prob:
        return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
    else:
        return (per_token_logps * loss_mask).sum(-1)


def concatenated_inputs(batch: Dict[str, Union[List, torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids', which are tensors of shape (batch_size, sequence_length).

    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1])
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('chosen', 'concatenated')
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0
            concatenated_key = k.replace('rejected', 'concatenated')
            concatenated_batch[concatenated_key] = torch.cat((
                concatenated_batch[concatenated_key],
                pad_to_length(batch[k], max_length, pad_value=pad_value),
            ), dim=0)
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.

           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.run_dir = run_dir
        self.device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')
        if config.loss.name == 'samdpo':
            torch.backends.cuda.matmul.allow_tf32 = False
            validate_samdpo(config, world_size=world_size)

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            dataset_path=config.get('dataset_path', None),
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
            seed=config.seed if config.loss.name == 'samdpo' else 0,
        )

        self.policy = policy
        self.reference_model = reference_model
        self.samdpo_regularizer = OutputGradientRegularizer(config.loss.sam, rank=rank) if config.loss.name == 'samdpo' else None

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs), noise_rate=config.noise_rate)
        rank0_print(f'Loaded train data iterator')
        eval_kwargs = dict(data_iterator_kwargs)
        # Use the same fixed eval subset for DPO and SAMDPO, regardless of train RNG.
        eval_kwargs['seed'] = config.seed
        if config.n_eval_examples is not None and (config.n_eval_examples <= 0 or
                config.n_eval_examples % config.eval_batch_size):
            raise ValueError('n_eval_examples must be a positive multiple of eval_batch_size, or null')
        self.eval_iterator = get_batch_iterator(**eval_kwargs, split='test', n_epochs=1, n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator)
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')
        eval_count = sum(len(batch['chosen_input_ids']) for batch in self.eval_batches)
        if config.n_eval_examples is not None and eval_count != config.n_eval_examples:
            raise ValueError(f'Requested {config.n_eval_examples} evaluation examples, but only '
                             f'{eval_count} are available as full batches; no training data is used')
        if any(name in ('hh', 'hh_pairwise_noise') for name in config.datasets):
            from preference_datasets import get_audited_hh_splits
            _, audit = get_audited_hh_splits(cache_dir=get_local_dir(config.local_dirs),
                                            dataset_path=config.get('dataset_path', None))
            audit = dict(audit, eval_examples=eval_count, eval_seed=config.seed,
                         world_size=world_size, eval_batch_size=config.eval_batch_size)
            rank0_print(f'HH dataset sizes and split audit: {audit}')
            if rank == 0:
                os.makedirs(self.run_dir, exist_ok=True)
                with open(os.path.join(self.run_dir, 'dataset_audit.json'), 'w', encoding='utf-8') as stream:
                    json.dump(audit, stream, ensure_ascii=False, indent=2)


    def get_batch_samples(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs."""

        # FSDP generation according to https://github.com/pytorch/pytorch/issues/100069
        ctx = lambda: (FSDP.summon_full_params(self.policy, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
        with ctx():
            policy_output = (self.policy.module if isinstance(self.policy, torch.nn.parallel.DistributedDataParallel) else self.policy).generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        if self.config.loss.name in {'dpo', 'ipo', 'samdpo'}:
            ctx = lambda: (FSDP.summon_full_params(self.reference_model, writeback=False, recurse=False) if 'FSDP' in self.config.trainer else contextlib.nullcontext())
            with ctx():
                reference_output = self.reference_model.generate(
                    batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'], max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name in {'dpo', 'ipo', 'samdpo'}:
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded

    def concatenated_forward(self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], return_aux: bool = False):
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.

           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        concatenated_batch = concatenated_inputs(batch)
        outputs = model(
            concatenated_batch['concatenated_input_ids'],
            attention_mask=concatenated_batch['concatenated_attention_mask'],
            output_hidden_states=return_aux,
            return_dict=True,
        )
        all_logits = outputs.logits.to(torch.float32)
        all_logps = _get_batch_logps(all_logits, concatenated_batch['concatenated_labels'], average_log_prob=False)
        chosen_logps = all_logps[:batch['chosen_input_ids'].shape[0]]
        rejected_logps = all_logps[batch['chosen_input_ids'].shape[0]:]
        if not return_aux:
            return chosen_logps, rejected_logps
        batch_size = batch['chosen_input_ids'].shape[0]
        hidden = outputs.hidden_states[-1]
        return chosen_logps, rejected_logps, {
            'chosen_hidden': hidden[:batch_size],
            'rejected_hidden': hidden[batch_size:],
            'chosen_logits': all_logits[:batch_size],
            'rejected_logits': all_logits[batch_size:],
            'chosen_labels': concatenated_batch['concatenated_labels'][:batch_size],
            'rejected_labels': concatenated_batch['concatenated_labels'][batch_size:],
        }


    def get_batch_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """Compute SFT, DPO, IPO, DrDPO, or standalone SAMDPO metrics."""
        metrics = {}
        train_test = 'train' if train else 'eval'
        samdpo_metrics = {}

        if loss_config.name in {'dpo', 'ipo', 'samdpo'}:
            policy_aux = loss_config.name == 'samdpo'
            policy_result = self.concatenated_forward(self.policy, batch, return_aux=policy_aux)
            if policy_aux:
                policy_chosen_logps, policy_rejected_logps, policy_extra = policy_result
            else:
                policy_chosen_logps, policy_rejected_logps = policy_result
            with torch.no_grad():
                reference_chosen_logps, reference_rejected_logps = self.concatenated_forward(self.reference_model, batch)

            if loss_config.name == 'samdpo':
                margins = (policy_chosen_logps - policy_rejected_logps) - (reference_chosen_logps - reference_rejected_logps)
                dpo_losses, chosen_rewards, rejected_rewards = preference_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps,
                    beta=loss_config.beta,
                    label_smoothing=loss_config.label_smoothing,
                    reference_free=loss_config.reference_free,
                )
                sam_penalty, basis_rank = samdpo_penalty(
                    policy_extra, margins, loss_config.beta, self.samdpo_regularizer,
                    self.batch_counter, train,
                )
                loss = dpo_losses.mean() + loss_config.sam.reg_weight * sam_penalty
                samdpo_metrics = {
                    f'loss/{train_test}/dpo': [dpo_losses.mean().detach().item()],
                    f'loss/{train_test}/sam_penalty': [sam_penalty.detach().item()],
                    f'loss/{train_test}/samdpo': [loss.detach().item()],
                    f'sam/{train_test}/basis_rank': [float(basis_rank)],
                }
                losses = dpo_losses
            else:
                if loss_config.name == 'dpo':
                    loss_kwargs = {'beta': loss_config.beta, 'reference_free': loss_config.reference_free, 'label_smoothing': loss_config.label_smoothing, 'ipo': False}
                else:
                    loss_kwargs = {'beta': loss_config.beta, 'ipo': True}
                losses, chosen_rewards, rejected_rewards = preference_loss(
                    policy_chosen_logps, policy_rejected_logps,
                    reference_chosen_logps, reference_rejected_logps, **loss_kwargs)
                loss = losses.mean()

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)
            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            # Explicit names distinguish implicit-reward accuracy from policy likelihood ranking.
            metrics[f'preference_acc/{train_test}'] = metrics[f'rewards_{train_test}/accuracies']
            policy_accuracies = (policy_chosen_logps > policy_rejected_logps).float().detach()
            metrics[f'policy_preference_acc/{train_test}'] = all_gather_if_needed(
                policy_accuracies, self.rank, self.world_size).cpu().tolist()
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()
            policy_rejected_logps = all_gather_if_needed(policy_rejected_logps.detach(), self.rank, self.world_size)
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_logps.cpu().numpy().tolist()
        elif loss_config.name == 'sft':
            policy_chosen_logits = self.policy(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_logps = _get_batch_logps(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)
            losses = -policy_chosen_logps
            loss = losses.mean()
            token_counts = (batch['chosen_labels'][:, 1:] != -100).sum(-1)
            metrics[f'tokens_{train_test}/response'] = all_gather_if_needed(
                token_counts.detach(), self.rank, self.world_size).cpu().tolist()
        else:
            raise ValueError(f'unknown loss {loss_config.name}')

        policy_chosen_logps = all_gather_if_needed(policy_chosen_logps.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_logps.cpu().numpy().tolist()
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()
        if samdpo_metrics:
            for key, value in samdpo_metrics.items():
                scalar = losses.new_tensor(value)
                metrics[key] = all_gather_if_needed(scalar, self.rank, self.world_size).cpu().tolist()
            metrics[f'loss/{train_test}'] = all_gather_if_needed(
                loss.detach().reshape(1), self.rank, self.world_size).cpu().tolist()
        if loss_config.name == 'samdpo':
            return loss, metrics
        if loss_config.mode_loss == "DrDPO":
            return -loss_config.mode_weight * torch.log(torch.mean(torch.exp(-losses / loss_config.mode_weight))), metrics
        return loss, metrics

    @staticmethod
    def summarize_metrics(metrics):
        result = {key: sum(values) / len(values) for key, values in metrics.items()}
        for split in ('train', 'eval'):
            key = f'tokens_{split}/response'
            if key in metrics:
                count = sum(metrics[key])
                if count <= 0:
                    raise ValueError('SFT batch has no response tokens')
                nll = sum(metrics[f'loss/{split}']) / count
                result[f'nll/{split}'] = nll
                result[f'perplexity/{split}'] = float(np.exp(nll))
        return result

    def record_metrics(self, phase, metrics):
        """Persist every update/evaluation independently of console/W&B throttling."""
        if self.rank != 0:
            return
        os.makedirs(self.run_dir, exist_ok=True)
        record = {'phase': phase, 'examples': self.example_counter,
                  'updates': self.batch_counter, 'time': time.time(), 'metrics': metrics}
        with open(os.path.join(self.run_dir, 'metrics.jsonl'), 'a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')

    def evaluate(self):
        """Evaluate the current weights and save metrics and optional generated samples."""
        rank0_print(f'Running evaluation after {self.example_counter} train examples')
        self.policy.eval()

        if not self.eval_batches:
            raise ValueError('No evaluation batches: increase n_eval_examples to at least eval_batch_size')
        all_eval_metrics = defaultdict(list)
        if self.config.sample_during_eval:
            all_policy_samples, all_reference_samples, all_prompts = [], [], []
            policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
            if self.config.loss.name in {'dpo', 'ipo', 'samdpo'}:
                reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

        for eval_batch in (tqdm.tqdm(self.eval_batches, desc='Computing eval metrics') if self.rank == 0 else self.eval_batches):
            local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.device)
            with torch.no_grad():
                _, eval_metrics = self.get_batch_metrics(local_eval_batch, self.config.loss, train=False)

            for k, v in eval_metrics.items():
                all_eval_metrics[k].extend(v)

        if self.config.sample_during_eval:
            if self.config.n_eval_model_samples < self.config.eval_batch_size:
                rank0_print(f'Warning: n_eval_model_samples ({self.config.n_eval_model_samples}) < eval_batch_size ({self.config.eval_batch_size}). Sampling from the first complete eval batch of prompts.')
                sample_batches = self.eval_batches[:1]
            else:
                n_sample_batches = self.config.n_eval_model_samples // self.config.eval_batch_size
                sample_batches = self.eval_batches[:n_sample_batches]
            for eval_batch in (tqdm.tqdm(sample_batches, desc='Generating samples...') if self.rank == 0 else sample_batches):
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.device)
                policy_samples, reference_samples = self.get_batch_samples(local_eval_batch)

                all_prompts.extend(eval_batch['prompt'])
                all_policy_samples.extend(policy_samples)
                all_reference_samples.extend(reference_samples)

                for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                    policy_text_table.add_data(self.example_counter, prompt, sample)
                if self.config.loss.name in {'dpo', 'ipo', 'samdpo'}:
                    for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                        reference_text_table.add_data(self.example_counter, prompt, sample)

        mean_eval_metrics = self.summarize_metrics(all_eval_metrics)
        mean_eval_metrics['eval/examples'] = len(all_eval_metrics['logps_eval/chosen'])
        self.record_metrics('eval', mean_eval_metrics)
        self.last_eval_metrics = mean_eval_metrics
        rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
        if self.config.sample_during_eval:
            rank0_print(json.dumps(all_policy_samples[:10], indent=2))
            if self.config.loss.name in {'dpo', 'ipo', 'samdpo'}:
                rank0_print(json.dumps(all_reference_samples[:10], indent=2))

        if self.config.wandb.enabled and self.rank == 0:
            wandb.log(mean_eval_metrics, step=self.example_counter)

            if self.config.sample_during_eval:
                wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                if self.config.loss.name in {'dpo', 'ipo', 'samdpo'}:
                    wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

        if self.rank == 0 and self.config.sample_during_eval:
            sample_dir = os.path.join(self.run_dir, 'samples')
            os.makedirs(sample_dir, exist_ok=True)
            with open(os.path.join(sample_dir, f'step-{self.example_counter}.json'), 'w', encoding='utf-8') as stream:
                json.dump({'examples': self.example_counter, 'prompts': all_prompts,
                           'policy_samples': all_policy_samples,
                           'reference_samples': all_reference_samples}, stream, ensure_ascii=False, indent=2)
        self.last_eval_counter = self.example_counter
        return mean_eval_metrics

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(
            [p for p in self.policy.parameters() if p.requires_grad], lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if self.config.loss.name in {'dpo', 'ipo', 'samdpo'}:
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None
        self.last_eval_counter = -1
        if self.config.loss.name == 'samdpo':
            from samdpo_runtime import restore_training
            restore_training(self)

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            if self.example_counter != self.last_eval_counter and self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
                mean_eval_metrics = self.evaluate()
                self.last_eval_counter = self.example_counter
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.device)
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.device)
                loss, metrics = self.get_batch_metrics(local_microbatch, self.config.loss, train=True)
                (loss / self.config.gradient_accumulation_steps).backward()

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm)
            batch_metrics['lr'].append(self.optimizer.param_groups[0]['lr'])

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            mean_train_metrics = self.summarize_metrics(batch_metrics)
            mean_train_metrics['counters/examples'] = self.example_counter
            mean_train_metrics['counters/updates'] = self.batch_counter
            self.record_metrics('train', mean_train_metrics)

            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####

        # Always assess the final weights before worker_main writes the sole checkpoint.
        if self.last_eval_counter != self.example_counter:
            self.evaluate()


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy."""
        return torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)

    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        if self.config.loss.name == 'samdpo':
            from samdpo_runtime import save_training
            return save_training(self, output_dir, metrics)
        policy_state_dict = self.policy.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        if self.config.loss.name == 'sft':
            return

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.

           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        mp_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mp_policy = MixedPrecision(param_dtype=mp_dtype, reduce_dtype=mp_dtype, buffer_dtype=mp_dtype)
        self.policy = FSDP(policy, **shared_fsdp_kwargs, mixed_precision=policy_mp_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                    CheckpointImpl,
                )
                non_reentrant_wrapper = functools.partial(
                    checkpoint_wrapper,
                    offload_to_cpu=False,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name in {'dpo', 'ipo', 'samdpo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)

        print('Loaded model on rank', rank)
        dist.barrier()

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()

    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        dist.barrier()
        # SFT is an initialization stage: retain only the final model, not optimizer state.
        if self.config.loss.name == 'sft':
            return

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()


class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)

        rank0_print('Sharding policy...')
        self.policy = tp.tensor_parallel(policy, sharded=True)
        if config.loss.name in {'dpo', 'ipo', 'samdpo'}:
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()

        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict



class DDPTrainer(BasicTrainer):
    """Ordinary data parallel SAMDPO: replicate the model, average only parameter gradients."""
    def __init__(self, policy, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        if config.loss.name != 'samdpo':
            raise ValueError('DDPTrainer is configured for SAMDPO')
        super().__init__(policy, config, seed, run_dir, reference_model, rank, world_size)
        device_ids = [rank] if self.device.type == 'cuda' else None
        self.policy = torch.nn.parallel.DistributedDataParallel(
            policy, device_ids=device_ids, broadcast_buffers=False, find_unused_parameters=False)
