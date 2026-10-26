import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn as nn
import transformers
from utils import get_local_dir, get_local_run_dir, disable_dropout, init_distributed, get_open_port
import os
import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import trainers
import wandb
import json
import socket
from typing import Optional, Set
try:
    import resource
except ImportError:
    resource = None


OmegaConf.register_new_resolver("get_local_run_dir", lambda exp_name, local_dirs: get_local_run_dir(exp_name, local_dirs))


def worker_main(rank: int, world_size: int, config: DictConfig, policy: nn.Module, reference_model: Optional[nn.Module] = None):
    """Main function for each worker process (may be only 1 for BasicTrainer/TensorParallelTrainer)."""
    if 'FSDP' in config.trainer or config.trainer == 'DDPTrainer':
        init_distributed(rank, world_size, port=config.fsdp_port,
                         backend='nccl' if torch.distributed.is_nccl_available() else 'gloo')
    
    if config.loss.name == 'samdpo':
        from samdpo_runtime import build_models
        torch.manual_seed(config.seed)
        policy, reference_model = build_models(config, rank, world_size)

    if config.debug:
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None

    if rank == 0 and config.wandb.enabled:
        os.environ['WANDB_CACHE_DIR'] = get_local_dir(config.local_dirs)
        wandb.init(
            entity=config.wandb.entity,
            project=config.wandb.project,
            config=OmegaConf.to_container(config),
            dir=get_local_dir(config.local_dirs),
            name=config.exp_name,
        )

    TrainerClass = getattr(trainers, config.trainer)
    print(f'Creating trainer on process {rank} with world size {world_size}')
    trainer = TrainerClass(policy, config, config.seed, config.local_run_dir, reference_model=reference_model, rank=rank, world_size=world_size)

    trainer.train()
    trainer.save(metrics=getattr(trainer, 'last_eval_metrics', None))
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    """Main entry point for training. Validates config, creates/initializes model(s), and kicks off worker process(es)."""

    # Resolve hydra references, e.g. so we don't re-compute the run directory
    OmegaConf.resolve(config)

    missing_keys: Set[str] = OmegaConf.missing_keys(config)
    if missing_keys:
        raise ValueError(f"Got missing keys in config:\n{missing_keys}")

    if config.loss.name == 'samdpo':
        from samdpo import validate_samdpo
        world_size = torch.cuda.device_count() if config.trainer == 'DDPTrainer' else 1
        if world_size < 1:
            raise ValueError('DDPTrainer requires at least one CUDA device')
        validate_samdpo(config, world_size=world_size)
        if config.eval_every < config.batch_size or config.eval_every % config.batch_size:
            raise ValueError('eval_every must be a positive multiple of batch_size')
        with open(os.path.join(config.local_run_dir, 'config.yaml'), 'w') as stream:
            OmegaConf.save(config, stream)
        if config.trainer == 'DDPTrainer':
            config.fsdp_port = config.fsdp_port or get_open_port()
            mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, None, None), join=True)
        else:
            worker_main(0, 1, config, None, None)
        return

    if config.trainer == 'DDPTrainer':
        raise ValueError('DDPTrainer supports SAMDPO only; use FSDPTrainer for four-GPU full-parameter SFT')
    world_size = torch.cuda.device_count() if 'FSDP' in config.trainer else 1
    divisor = world_size * config.gradient_accumulation_steps
    if world_size < 1 or config.gradient_accumulation_steps < 1 or config.batch_size < divisor or config.batch_size % divisor:
        raise ValueError('batch_size must be a positive multiple of world_size * gradient_accumulation_steps')
    if config.eval_batch_size < world_size or config.eval_batch_size % world_size:
        raise ValueError('eval_batch_size must be a positive multiple of world_size')
    if config.eval_every < config.batch_size:
        raise ValueError('eval_every must be at least batch_size (measured in examples)')

    if config.eval_every % config.batch_size != 0:
        print('WARNING: eval_every must be divisible by batch_size')
        print('Setting eval_every to', config.eval_every - config.eval_every % config.batch_size)
        config.eval_every = config.eval_every - config.eval_every % config.batch_size

    if 'FSDP' in config.trainer and config.fsdp_port is None:
        free_port = get_open_port()
        print('no FSDP port specified; using open port for FSDP:', free_port)
        config.fsdp_port = free_port

    print(OmegaConf.to_yaml(config))

    config_path = os.path.join(config.local_run_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        OmegaConf.save(config, f)

    print('=' * 80)
    print(f'Writing to {socket.gethostname()}:{config.local_run_dir}')
    print('=' * 80)
 
    os.environ['XDG_CACHE_HOME'] = get_local_dir(config.local_dirs)
    print('building policy')
    model_kwargs = {'device_map': 'balanced'} if config.trainer == 'BasicTrainer' else {}
    policy_dtype = getattr(torch, config.model.policy_dtype)
    policy = transformers.AutoModelForCausalLM.from_pretrained(
        config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=policy_dtype, **model_kwargs)
    disable_dropout(policy)
    if config.activation_checkpointing:
        policy.config.use_cache = False

    if config.loss.name in {'dpo', 'ipo', 'samdpo'}:
        print('building reference model')
        reference_model_dtype = getattr(torch, config.model.reference_dtype)
        reference_model = transformers.AutoModelForCausalLM.from_pretrained(
            config.model.name_or_path, cache_dir=get_local_dir(config.local_dirs), low_cpu_mem_usage=True, torch_dtype=reference_model_dtype, **model_kwargs)
        disable_dropout(reference_model)
    else:
        reference_model = None

    if config.model.archive is not None:
        state_dict = torch.load(config.model.archive, map_location='cpu')
        step, metrics = state_dict['step_idx'], state_dict['metrics']
        print(f'loading pre-trained weights at step {step} from {config.model.archive} with metrics {json.dumps(metrics, indent=2)}')
        policy.load_state_dict(state_dict['state'])
        if config.loss.name in {'dpo', 'ipo', 'samdpo'}:
            reference_model.load_state_dict(state_dict['state'])
        print('loaded pre-trained weights')
    
    if 'FSDP' in config.trainer:
        world_size = torch.cuda.device_count()
        print('starting', world_size, 'processes for FSDP training')
        if resource is not None:
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
            print(f'setting RLIMIT_NOFILE soft limit to {hard} from {soft}')
        mp.spawn(worker_main, nprocs=world_size, args=(world_size, config, policy, reference_model), join=True)
    else:
        print('starting single-process worker')
        worker_main(0, 1, config, policy, reference_model)


if __name__ == '__main__':
    main()
