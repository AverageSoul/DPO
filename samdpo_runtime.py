"""Model construction and per-rank exact-state checkpoints for SAMDPO."""
import os
import random
import numpy as np
import torch
from omegaconf import OmegaConf
from samdpo import prepare_transformer, validate_samdpo


def signature(config):
    raw = OmegaConf.to_container(config, resolve=True)
    result = {key: raw[key] for key in ['model', 'datasets', 'noise_rate', 'seed',
            'batch_size', 'eval_batch_size', 'max_length', 'max_prompt_length',
            'n_epochs', 'n_examples', 'n_eval_examples', 'optimizer', 'lr',
            'warmup_steps', 'max_grad_norm', 'eval_every', 'sample_during_eval',
            'n_eval_model_samples', 'do_first_eval']}
    result['dataset_path'] = raw.get('dataset_path')
    result['hh_split_policy_version'] = 1
    result['parameterization'] = 'transformer_fp32_frozen_output_head'
    result['loss'] = {key: value for key, value in raw['loss'].items() if key != 'resume'}
    return result


def build_models(config, rank, world_size):
    import transformers
    from utils import disable_dropout, get_local_dir
    device = torch.device('cuda', rank) if torch.cuda.is_available() else torch.device('cpu')
    cache = get_local_dir(config.local_dirs)
    kwargs = dict(cache_dir=cache, low_cpu_mem_usage=True)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, torch_dtype=torch.float32, **kwargs)
    reference = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, torch_dtype=getattr(torch, config.model.reference_dtype), **kwargs)
    if config.model.archive is not None:
        state = torch.load(config.model.archive, map_location='cpu', weights_only=False)['state']
        policy.load_state_dict(state)
        reference.load_state_dict(state)
        del state
    validate_samdpo(config, policy.get_output_embeddings().weight.shape[1], world_size)
    reference.requires_grad_(False)
    reference.eval()
    disable_dropout(reference)
    policy = prepare_transformer(policy)
    policy.config.use_cache = False
    disable_dropout(policy)
    if config.loss.resume:
        common = torch.load(os.path.join(config.loss.resume, 'samdpo.pt'), map_location='cpu', weights_only=False)
        if common.get('format_version') != 2:
            raise ValueError('Resume requires a full-Transformer SAMDPO checkpoint (format_version=2)')
        if common['signature'] != signature(config) or common['world_size'] != world_size:
            raise ValueError('Resume requires identical model, data, loss, optimizer and world size')
        policy.load_state_dict(common['state'], strict=True)
    return policy.to(device), reference.to(device)


def unwrap(policy):
    from torch.nn.parallel import DistributedDataParallel
    return policy.module if isinstance(policy, DistributedDataParallel) else policy


def save_training(trainer, output_dir=None, metrics=None):
    directory = output_dir or os.path.join(trainer.run_dir, 'LATEST')
    os.makedirs(directory, exist_ok=True)
    policy = unwrap(trainer.policy)
    if trainer.rank == 0:
        common = dict(format_version=2, world_size=trainer.world_size,
                      signature=signature(trainer.config), batch_counter=trainer.batch_counter,
                      state={name: value.detach().cpu() for name, value in policy.state_dict().items()},
                      metrics=metrics or {})
        torch.save(common, os.path.join(directory, 'samdpo.pt.tmp'))
        os.replace(os.path.join(directory, 'samdpo.pt.tmp'), os.path.join(directory, 'samdpo.pt'))
    local = dict(batch_counter=trainer.batch_counter, example_counter=trainer.example_counter,
                 last_eval_counter=trainer.last_eval_counter,
                 regularizer=trainer.samdpo_regularizer.state_dict(),
                 optimizer=trainer.optimizer.state_dict(), scheduler=trainer.scheduler.state_dict(),
                 torch_rng=torch.get_rng_state(), numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                 cuda_rng=torch.cuda.get_rng_state(trainer.device) if trainer.device.type == 'cuda' else None)
    destination = os.path.join(directory, 'rank-%d.pt' % trainer.rank)
    torch.save(local, destination + '.tmp')
    os.replace(destination + '.tmp', destination)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()


def restore_training(trainer):
    directory = trainer.config.loss.resume
    if not directory:
        return
    local = torch.load(os.path.join(directory, 'rank-%d.pt' % trainer.rank), map_location='cpu', weights_only=False)
    common = torch.load(os.path.join(directory, 'samdpo.pt'), map_location='cpu', weights_only=False)
    if local['batch_counter'] != common['batch_counter']:
        raise ValueError('Incomplete checkpoint: common and rank training steps differ')
    trainer.optimizer.load_state_dict(local['optimizer'])
    trainer.scheduler.load_state_dict(local['scheduler'])
    trainer.samdpo_regularizer.load_state_dict(local['regularizer'], trainer.device)
    trainer.batch_counter, trainer.example_counter = local['batch_counter'], local['example_counter']
    trainer.last_eval_counter = local['last_eval_counter']
    # Reconstruct the deterministic iterator and consume exactly the completed batches.
    for _ in range(trainer.batch_counter):
        try:
            next(trainer.train_iterator)
        except StopIteration:
            raise ValueError('Resume data iterator ended before saved progress')
    torch.set_rng_state(local['torch_rng'])
    np.random.set_state(local['numpy_rng'])
    random.setstate(local['python_rng'])
    if local['cuda_rng'] is not None:
        torch.cuda.set_rng_state(local['cuda_rng'], trainer.device)
