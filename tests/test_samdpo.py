import copy
import os
import random
import socket
import sys
import tempfile
import unittest
from types import SimpleNamespace as NS
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
import torch.nn.functional as F
from samdpo import (OutputGradientRegularizer, projected_gradient, rank_aware_orth,
                    samdpo_penalty, prepare_transformer, validate_samdpo)


def settings(**kwargs):
    values = dict(seed=7, sketch_dim=3, refresh_interval=2, orth_eps=1e-6, reg_weight=0.2)
    values.update(kwargs)
    return NS(**values)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('features', torch.randn(4, 2, 5))
        self.register_buffer('head', torch.randn(11, 5))
        self.a = torch.nn.Parameter(torch.randn(2, 5) * 0.1)
        self.b = torch.nn.Parameter(torch.randn(5, 2) * 0.1)

    def forward(self, index):
        x = self.features[index]
        hidden = x + x @ self.a.T @ self.b.T
        return hidden, hidden @ self.head.T


def objective(model, rank, regularizer):
    h, z = model(torch.tensor([rank * 2, rank * 2 + 1]))
    targets = torch.tensor([[1, 2], [3, 4]])
    p = torch.softmax(z.float(), -1)
    scores = z.float().log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).sum(-1)
    margin = scores[0] - scores[1] - 0.2
    loss = F.softplus(-0.4 * margin)
    alpha = -0.4 * torch.sigmoid(-0.4 * margin)
    coefficients = torch.stack((alpha.expand(2), -alpha.expand(2))).reshape(-1)
    penalty, _ = regularizer(h.reshape(-1, 5), p.reshape(-1, 11), targets.reshape(-1), coefficients, 0)
    return loss + 0.2 * penalty


def ddp_worker(rank, port, destination):
    torch.set_num_threads(1)
    os.environ.setdefault('USE_LIBUV', '0')
    torch.distributed.init_process_group('gloo', init_method='tcp://127.0.0.1:%d' % port,
                                         rank=rank, world_size=2)
    torch.manual_seed(20)
    model = Toy()
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    reg = OutputGradientRegularizer(settings(), rank)
    objective(ddp, rank, reg).backward()
    # Every rank must obtain the same average of the two LOCAL penalties' gradients.
    torch.save([p.grad for p in model.parameters()], os.path.join(destination, '%d.pt' % rank))
    torch.distributed.destroy_process_group()


class SAMDPOTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20)
        torch.set_num_threads(1)

    def test_analytic_gradient_and_parameter_derivatives(self):
        model = Toy()
        hidden, _ = model(torch.tensor([0, 1]))
        virtual_head = model.head.clone().requires_grad_(True)
        logits = hidden @ virtual_head.T
        targets = torch.tensor([[1, 2], [3, 4]])
        scores = logits.log_softmax(-1).gather(-1, targets[..., None]).squeeze(-1).sum(-1)
        margin = scores[0] - scores[1] - 0.2
        loss = F.softplus(-0.4 * margin)
        g = torch.autograd.grad(loss, virtual_head, create_graph=True)[0]
        c = rank_aware_orth(torch.randn(5, 3)).detach()
        alpha = -0.4 * torch.sigmoid(-0.4 * margin)
        coeff = torch.stack([alpha.expand(2), -alpha.expand(2)]).reshape(-1)
        analytic = projected_gradient(hidden.reshape(-1,5), logits.softmax(-1).reshape(-1,11),
                                      targets.reshape(-1), coeff, c)
        torch.testing.assert_close(analytic, g @ c, atol=2e-6, rtol=2e-5)
        pa = 0.5 * analytic.square().sum()
        pb = 0.5 * (g @ c).square().sum()
        ga = torch.autograd.grad(pa, (model.a, model.b), retain_graph=True)
        gb = torch.autograd.grad(pb, (model.a, model.b))
        for x, y in zip(ga, gb):
            torch.testing.assert_close(x, y, atol=2e-6, rtol=3e-5)
            self.assertGreater(x.norm().item(), 0)

    def test_cancellation_zero_rank_refresh_eval_and_state(self):
        h = torch.randn(1, 5).repeat(2, 1).requires_grad_()
        p = torch.softmax(torch.randn(1,11), -1).repeat(2,1)
        targets = torch.tensor([1,1])
        coefficients = torch.tensor([0.1,-0.1])
        reg = OutputGradientRegularizer(settings())
        penalty, rank = reg(h, p, targets, coefficients, 0)
        self.assertEqual(rank, 0)
        self.assertEqual(penalty.item(), 0)
        penalty.backward()
        torch.testing.assert_close(h.grad, torch.zeros_like(h))
        reg = OutputGradientRegularizer(settings(), rank=1)
        reg(h, p, targets, torch.ones(2), 0)
        basis = reg.basis.clone()
        reg(h * 2, p, targets, torch.ones(2), 1)
        torch.testing.assert_close(reg.basis, basis)
        reg(h, p, targets, torch.ones(2), 9, refresh=False)
        self.assertEqual(reg.last_refresh, 0)
        reg(h, p, targets, torch.ones(2), 2)
        self.assertEqual(reg.last_refresh, 2)
        loaded = OutputGradientRegularizer(settings(), rank=1)
        loaded.load_state_dict(reg.state_dict(), 'cpu')
        torch.testing.assert_close(loaded.basis, reg.basis)
        with self.assertRaises(ValueError):
            OutputGradientRegularizer(settings(), rank=0).load_state_dict(reg.state_dict(), 'cpu')
        empty = OutputGradientRegularizer(settings())
        empty(h, p, targets, coefficients, 0, refresh=False)
        self.assertIsNone(empty.basis)

    def test_low_precision_and_outer_autocast(self):
        for dtype in (torch.bfloat16, torch.float16):
            h = torch.randn(4,5).to(dtype).requires_grad_()
            logits = torch.randn(4,11).requires_grad_()
            reg = OutputGradientRegularizer(settings())
            with torch.autocast('cpu', dtype=torch.bfloat16):
                penalty, _ = reg(h, logits.softmax(-1), torch.tensor([1,2,3,4]),
                                 torch.ones(4, dtype=dtype), 0)
            self.assertEqual(penalty.dtype, torch.float32)
            penalty.backward()
            self.assertTrue(torch.isfinite(h.grad).all())
            self.assertTrue(torch.isfinite(logits.grad).all())

    def test_transformer_freezing_and_real_trainer_loss(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from trainers import BasicTrainer
        from utils import disable_dropout
        policy = GPT2LMHeadModel(GPT2Config(vocab_size=19, n_embd=8, n_layer=1, n_head=2,
                                           n_positions=16, resid_pdrop=0, embd_pdrop=0, attn_pdrop=0))
        reference = copy.deepcopy(policy).requires_grad_(False).eval()
        policy = prepare_transformer(policy)
        self.assertTrue(all(parameter.requires_grad for parameter in policy.transformer.h.parameters()))
        self.assertFalse(any('lora_' in name for name, _ in policy.named_parameters()))
        disable_dropout(policy)
        self.assertIs(policy.get_output_embeddings().weight, policy.get_input_embeddings().weight)
        self.assertFalse(policy.get_output_embeddings().weight.requires_grad)
        trainer = BasicTrainer.__new__(BasicTrainer)
        trainer.policy, trainer.reference_model = policy, reference
        trainer.rank, trainer.world_size, trainer.batch_counter = 0, 1, 0
        trainer.samdpo_regularizer = OutputGradientRegularizer(settings())
        batch = {}
        for branch in ['chosen','rejected']:
            ids = torch.randint(0,19,(2,6))
            labels = ids.clone(); labels[:, :2] = -100
            batch[branch+'_input_ids'] = ids
            batch[branch+'_attention_mask'] = torch.ones_like(ids)
            batch[branch+'_labels'] = labels
        cfg = NS(name='samdpo', beta=0.4, label_smoothing=0, reference_free=False, sam=settings())
        loss, metrics = trainer.get_batch_metrics(batch,cfg,True)
        pc, pr, aux = trainer.concatenated_forward(policy,batch,True)
        rc, rr = trainer.concatenated_forward(reference,batch)
        penalty, _ = samdpo_penalty(aux, pc-pr-rc+rr, cfg.beta, trainer.samdpo_regularizer, 0, True)
        grads = torch.autograd.grad(penalty, [p for p in policy.parameters() if p.requires_grad])
        self.assertTrue(any(g.norm() > 0 for g in grads))
        cfg.sam.reg_weight = 0
        plain, _ = trainer.get_batch_metrics(batch,cfg,True)
        torch.testing.assert_close(plain, F.softplus(-cfg.beta * (pc-pr-rc+rr)).mean())
        with torch.no_grad():
            trainer.get_batch_metrics(batch,cfg,False)
        self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_untied_input_embeddings_train_and_frozen_head_stays_fixed(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        policy = prepare_transformer(GPT2LMHeadModel(GPT2Config(
            vocab_size=19, n_embd=8, n_layer=1, n_head=2, n_positions=16,
            tie_word_embeddings=False)))
        self.assertTrue(policy.get_input_embeddings().weight.requires_grad)
        before = policy.get_output_embeddings().weight.detach().clone()
        optimizer = torch.optim.SGD([p for p in policy.parameters() if p.requires_grad], lr=0.01)
        ids = torch.randint(0, 19, (2, 6))
        policy(ids, labels=ids).loss.backward()
        self.assertGreater(policy.get_input_embeddings().weight.grad.norm().item(), 0)
        self.assertTrue(all(p.grad is not None for p in policy.transformer.h.parameters()))
        optimizer.step()
        torch.testing.assert_close(before, policy.get_output_embeddings().weight, atol=0, rtol=0)

    def test_model_build_transformer_resume_keeps_original_reference(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from omegaconf import OmegaConf
        from samdpo_runtime import build_models, save_training
        keys = ['datasets', 'noise_rate', 'seed', 'batch_size', 'eval_batch_size',
                'max_length', 'max_prompt_length', 'n_epochs', 'n_examples', 'n_eval_examples',
                'optimizer', 'lr', 'warmup_steps', 'max_grad_norm', 'eval_every',
                'sample_during_eval', 'n_eval_model_samples', 'do_first_eval']
        with tempfile.TemporaryDirectory() as directory:
            base = GPT2LMHeadModel(GPT2Config(vocab_size=19,n_embd=8,n_layer=1,n_head=2,n_positions=16))
            base.save_pretrained(directory)
            cfg = OmegaConf.create({key: None for key in keys})
            cfg.trainer='BasicTrainer'; cfg.activation_checkpointing=False
            cfg.gradient_accumulation_steps=1; cfg.batch_size=2; cfg.eval_batch_size=2
            cfg.seed=0; cfg.local_dirs=[directory]
            cfg.model=dict(name_or_path=directory,archive=None,policy_dtype='float32',reference_dtype='float32')
            cfg.loss=dict(name='samdpo',resume=None,beta=0.4,label_smoothing=0,reference_free=False,
                          sam=vars(settings()))
            policy,reference=build_models(cfg,0,1)
            with torch.no_grad():
                for p in policy.parameters():
                    if p.requires_grad: p.add_(0.3)
            t=NS(policy=policy,reference_model=reference,config=cfg,rank=0,world_size=1,
                 device=next(policy.parameters()).device,batch_counter=0,example_counter=0,last_eval_counter=-1,
                 samdpo_regularizer=OutputGradientRegularizer(cfg.loss.sam))
            t.optimizer=torch.optim.Adam([p for p in policy.parameters() if p.requires_grad],lr=0.01)
            t.scheduler=torch.optim.lr_scheduler.LambdaLR(t.optimizer,lambda _:1)
            checkpoint=os.path.join(directory,'checkpoint')
            save_training(t,checkpoint)
            cfg.loss.resume=checkpoint
            restored,ref2=build_models(cfg,0,1)
            for (n,p),(n2,q) in zip(policy.named_parameters(),restored.named_parameters()):
                self.assertEqual(n,n2); torch.testing.assert_close(p,q)
            for p,q in zip(reference.parameters(),ref2.parameters()): torch.testing.assert_close(p,q)
            self.assertTrue(all(not p.requires_grad for p in ref2.parameters()))
            checkpoint_state = torch.load(os.path.join(checkpoint, 'samdpo.pt'), weights_only=False)
            self.assertEqual(checkpoint_state['format_version'], 2)
            self.assertEqual(set(checkpoint_state['state']), set(policy.state_dict()))

    def test_full_basic_trainer_checkpoint_continuation(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from omegaconf import OmegaConf
        from unittest.mock import patch
        from trainers import BasicTrainer
        from samdpo_runtime import build_models
        with tempfile.TemporaryDirectory() as directory:
            base = GPT2LMHeadModel(GPT2Config(vocab_size=19,n_embd=8,n_layer=1,n_head=2,n_positions=16))
            base.save_pretrained(directory)
            cfg=OmegaConf.create(dict(
                trainer='BasicTrainer',activation_checkpointing=False,gradient_accumulation_steps=1,
                seed=0,datasets=['toy'],noise_rate=0,batch_size=2,eval_batch_size=2,
                max_length=6,max_prompt_length=2,n_epochs=None,n_examples=4,n_eval_examples=2,
                optimizer='Adam',lr=0.01,warmup_steps=0,max_grad_norm=1.0,eval_every=2,
                sample_during_eval=False,n_eval_model_samples=0,do_first_eval=True,
                minimum_log_interval_secs=0,debug=False,wandb=dict(enabled=False),local_dirs=[directory],
                model=dict(name_or_path=directory,tokenizer_name_or_path=None,archive=None,
                           policy_dtype='float32',reference_dtype='float32'),
                loss=dict(name='samdpo',resume=None,beta=0.4,label_smoothing=0,reference_free=False,
                          sam=vars(settings()))))
            batches=[]
            for step in range(2):
                batch={}
                for branch in ['chosen','rejected']:
                    ids=torch.randint(0,19,(2,6)); labels=ids.clone(); labels[:,:2]=-100
                    batch[branch+'_input_ids']=ids; batch[branch+'_labels']=labels
                    batch[branch+'_attention_mask']=torch.ones_like(ids)
                batches.append(batch)
            def iterator(**kwargs):
                return iter(batches if kwargs['split']=='train' else batches[:1])
            with patch('trainers.get_batch_iterator',side_effect=iterator), patch(
                    'trainers.transformers.AutoTokenizer.from_pretrained',return_value=NS(pad_token_id=0)):
                torch.manual_seed(0)
                policy,ref=build_models(cfg,0,1)
                original=BasicTrainer(policy,cfg,0,directory,ref)
                # Exercise an explicit mid-run save; the production loop never saves periodically.
                evaluate = original.evaluate
                def evaluate_and_save_for_test():
                    metrics = evaluate()
                    if original.example_counter == 2:
                        original.save(os.path.join(directory, 'manual-step-2'), metrics)
                    return metrics
                with patch.object(original, 'evaluate', side_effect=evaluate_and_save_for_test):
                    original.train()
                cfg.loss.resume=os.path.join(directory,'manual-step-2')
                torch.manual_seed(0)
                policy2,ref2=build_models(cfg,0,1)
                resumed=BasicTrainer(policy2,cfg,0,directory,ref2)
                resumed.train()
                self.assertEqual(resumed.batch_counter,original.batch_counter)
                for p,q in zip(original.policy.parameters(),resumed.policy.parameters()):
                    torch.testing.assert_close(p,q,atol=0,rtol=0)
                torch.testing.assert_close(original.samdpo_regularizer.basis,resumed.samdpo_regularizer.basis,
                                           atol=0,rtol=0)

    def test_checkpoint_resume_next_update_and_rng(self):
        from omegaconf import OmegaConf
        from samdpo_runtime import save_training, restore_training
        keys = ['model', 'datasets', 'noise_rate', 'seed', 'batch_size', 'eval_batch_size',
                'max_length', 'max_prompt_length', 'n_epochs', 'n_examples', 'n_eval_examples',
                'optimizer', 'lr', 'warmup_steps', 'max_grad_norm', 'eval_every',
                'sample_during_eval', 'n_eval_model_samples', 'do_first_eval']
        cfg = OmegaConf.create({key: None for key in keys})
        cfg.loss = dict(name='samdpo', resume=None, sam=vars(settings()))
        def trainer(model, config):
            t = NS(policy=model, config=config, rank=0, world_size=1, device=torch.device('cpu'),
                   samdpo_regularizer=OutputGradientRegularizer(settings()), batch_counter=0,
                   example_counter=0, last_eval_counter=-1, train_iterator=iter(range(10)))
            t.optimizer = torch.optim.Adam(model.parameters(),lr=0.01)
            t.scheduler = torch.optim.lr_scheduler.LambdaLR(t.optimizer,lambda _: 1)
            return t
        t = trainer(Toy(),cfg)
        initial = copy.deepcopy(t.policy)
        objective(t.policy,0,t.samdpo_regularizer).backward()
        t.optimizer.step(); t.scheduler.step(); t.optimizer.zero_grad()
        t.batch_counter, t.example_counter = 1, 2
        with tempfile.TemporaryDirectory() as directory:
            save_training(t,directory)
            expected_rng = (torch.rand(3),np.random.rand(3),random.random())
            objective(t.policy,0,t.samdpo_regularizer).backward()
            t.optimizer.step()
            resumed_cfg = copy.deepcopy(cfg); resumed_cfg.loss.resume = directory
            resumed = trainer(initial,resumed_cfg)
            common=torch.load(os.path.join(directory,'samdpo.pt'),weights_only=False)
            resumed.policy.load_state_dict(common['state'])
            restore_training(resumed)
            torch.testing.assert_close(torch.rand(3),expected_rng[0])
            np.testing.assert_equal(np.random.rand(3),expected_rng[1])
            self.assertEqual(random.random(),expected_rng[2])
            self.assertEqual(next(resumed.train_iterator),1)
            self.assertEqual(resumed.batch_counter,1)
            objective(resumed.policy,0,resumed.samdpo_regularizer).backward()
            resumed.optimizer.step()
            for a,b in zip(t.policy.parameters(),resumed.policy.parameters()):
                torch.testing.assert_close(a,b,atol=0,rtol=0)

    def test_config_boundaries(self):
        cfg = NS(trainer='BasicTrainer', gradient_accumulation_steps=1, activation_checkpointing=False,
                 batch_size=4, eval_batch_size=4,
                 loss=NS(beta=0.1,label_smoothing=0,reference_free=False,sam=settings()))
        validate_samdpo(cfg, hidden_dim=5)
        legacy = copy.deepcopy(cfg)
        legacy.loss.lora = NS(r=32)
        with self.assertRaises(ValueError): validate_samdpo(legacy,5)
        for key, value in [('trainer','FSDPTrainer'),('gradient_accumulation_steps',2),('batch_size',0)]:
            c=copy.deepcopy(cfg); setattr(c,key,value)
            with self.assertRaises(ValueError): validate_samdpo(c,5)
        with self.assertRaises(ValueError): validate_samdpo(cfg,2)
        cfg.loss.reference_free=True
        with self.assertRaises(ValueError): validate_samdpo(cfg,5)

    def test_two_process_ddp_matches_local_objective_average(self):
        serial = Toy()
        loss = sum(objective(serial, rank, OutputGradientRegularizer(settings(),rank)) for rank in range(2))/2
        loss.backward()
        expected = [p.grad for p in serial.parameters()]
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0)); port=sock.getsockname()[1]
        with tempfile.TemporaryDirectory() as directory:
            torch.multiprocessing.spawn(ddp_worker,args=(port,directory),nprocs=2,join=True)
            for rank in range(2):
                actual = torch.load(os.path.join(directory,'%d.pt'%rank),weights_only=True)
                for x,y in zip(actual,expected):
                    torch.testing.assert_close(x,y,atol=3e-6,rtol=3e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
