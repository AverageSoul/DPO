"""PDF SAMDPO: local output-mapping gradient projection, with a detached basis."""
import math
import torch


def rank_aware_orth(matrix, relative_tolerance=1e-6):
    left, values, _ = torch.linalg.svd(matrix.float(), full_matrices=False)
    if values.numel() == 0:
        return matrix.float()[:, :0]
    return left[:, values > relative_tolerance * values.max()]


def projected_gradient(hidden, probabilities, targets, coefficients, basis):
    # Casts preserve the graph; only basis is a stop-gradient quantity.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        hidden, probabilities, coefficients = hidden.float(), probabilities.float(), coefficients.float()
        tokens = coefficients[:, None] * (hidden @ basis.detach().float())
        target_term = probabilities.new_zeros((probabilities.shape[1], basis.shape[1]))
        return target_term.index_add(0, targets, tokens) - probabilities.T @ tokens


class OutputGradientRegularizer:
    def __init__(self, config, rank=0):
        self.config, self.rank = config, rank
        self.basis, self.last_refresh = None, -1

    def __call__(self, hidden, probabilities, targets, coefficients, step, refresh=True):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            hidden, probabilities, coefficients = hidden.float(), probabilities.float(), coefficients.float()
            due = self.basis is None or step - self.last_refresh >= self.config.refresh_interval
            if refresh and due:
                with torch.no_grad():
                    generator = torch.Generator(device=hidden.device)
                    # Rank and refresh index have separate streams; do not touch global RNG.
                    generator.manual_seed(self.config.seed + self.rank * 1000003 + step)
                    omega = torch.randn(probabilities.shape[1], self.config.sketch_dim,
                                        generator=generator, device=hidden.device, dtype=torch.float32)
                    sketch = hidden.detach().T @ (coefficients.detach()[:, None] *
                              (omega[targets] - probabilities.detach() @ omega))
                    self.basis = rank_aware_orth(sketch, self.config.orth_eps).detach()
                    self.last_refresh = step
            # Evaluation never initializes or changes the training basis.
            basis = self.basis
            if basis is None:
                basis = hidden.new_zeros((hidden.shape[1], 0))
            gradient = projected_gradient(hidden, probabilities, targets, coefficients, basis)
            return 0.5 * gradient.square().sum(), basis.shape[1]

    def state_dict(self):
        return {'basis': None if self.basis is None else self.basis.detach().cpu(),
                'last_refresh': self.last_refresh, 'rank': self.rank}

    def load_state_dict(self, state, device):
        if state['rank'] != self.rank:
            raise ValueError('Cannot restore another rank\'s local SAMDPO basis')
        self.basis = None if state['basis'] is None else state['basis'].to(device=device, dtype=torch.float32).detach()
        self.last_refresh = state['last_refresh']


def samdpo_penalty(aux, margins, beta, regularizer, step, train):
    hidden, logits, targets, coefficients = [], [], [], []
    # This is the derivative of the LOCAL mean DPO loss, including 1/B.
    alpha = -beta * torch.sigmoid(-beta * margins.float()) / margins.numel()
    for branch, sign in [('chosen', 1), ('rejected', -1)]:
        labels = aux[branch + '_labels'][:, 1:]
        mask = labels != -100
        hidden.append(aux[branch + '_hidden'][:, :-1][mask])
        logits.append(aux[branch + '_logits'][:, :-1][mask])
        targets.append(labels[mask])
        coefficients.append((sign * alpha[:, None]).expand_as(labels)[mask])
    h, z = torch.cat(hidden), torch.cat(logits)
    with torch.autocast(device_type=h.device.type, enabled=False):
        probabilities = torch.softmax(z.float(), dim=-1)
        return regularizer(h, probabilities, torch.cat(targets), torch.cat(coefficients), step, refresh=train)


def validate_samdpo(config, hidden_dim=None, world_size=1):
    if config.trainer not in ('BasicTrainer', 'DDPTrainer'):
        raise ValueError('SAMDPO supports single-device BasicTrainer and ordinary DDPTrainer only')
    if config.gradient_accumulation_steps != 1:
        raise ValueError('SAMDPO requires gradient_accumulation_steps=1')
    if config.loss.label_smoothing != 0 or config.loss.reference_free:
        raise ValueError('SAMDPO requires label_smoothing=0 and reference_free=false')
    if config.activation_checkpointing:
        raise ValueError('SAMDPO retains the full forward graph; activation_checkpointing must be false')
    if config.trainer == 'BasicTrainer' and world_size != 1:
        raise ValueError('BasicTrainer is single-device for SAMDPO')
    for key in ('batch_size', 'eval_batch_size'):
        value = getattr(config, key)
        if not isinstance(value, int) or value < world_size or value % world_size:
            raise ValueError(key + ' must be positive and divisible by world size')
    sam = config.loss.sam
    for key in ('sketch_dim', 'refresh_interval'):
        if not isinstance(getattr(sam, key), int) or getattr(sam, key) < 1:
            raise ValueError(key + ' must be a positive integer')
    if hidden_dim is not None and sam.sketch_dim > hidden_dim:
        raise ValueError('sketch_dim must not exceed hidden dimension')
    for value, name, allow_zero in [(sam.reg_weight, 'reg_weight', True),
                                    (sam.orth_eps, 'orth_eps', False),
                                    (config.loss.beta, 'beta', False)]:
        if not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
            raise ValueError(name + ' is outside its valid range')
    if getattr(config.loss, 'lora', None) is not None:
        raise ValueError('SAMDPO trains Transformer parameters; remove loss.lora overrides')


def prepare_transformer(policy):
    policy.float().requires_grad_(True)
    head = policy.get_output_embeddings()
    if head is None:
        raise ValueError('SAMDPO requires an output head')
    head.requires_grad_(False)
    if not any(parameter.requires_grad for parameter in policy.parameters()):
        raise ValueError('SAMDPO requires trainable Transformer parameters')
    return policy
