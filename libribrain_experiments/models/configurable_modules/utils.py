import torch
import torch.nn.functional as F
from torch.nn import Conv1d, ELU
from torch.nn import Softsign, GRU, Linear, ReLU, Sigmoid, GELU, BatchNorm1d
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from libribrain_experiments.models.util_layers import Permute, LSTMBlock, SelectChannels
from libribrain_experiments.models.util_layers import FlattenTime
from libribrain_experiments.models.util_losses import BCEWithLogitsLossWithSmoothing
from torch.nn import LayerNorm
from libribrain_experiments.models.average_groups import AverageGroups
from libribrain_experiments.models.dyslexnet import DyslexNetTransformer
from libribrain_experiments.models.bertspeech import BertSpeech
from libribrain_experiments.models.conformer import ConformerSpeech
from libribrain_experiments.models.conformer_seq2seq import ConformerSeq2Seq
from libribrain_experiments.models.megnet import MEGNet
from libribrain_experiments.models.neural_ensemble import NeuralEnsemble


class BalancedSoftmaxCE(nn.Module):
    """
    Implements:  -log( exp(z_y) / sum_j n_j * exp(z_j) )
    where n_j are class counts. Works well on long-tail.
    Supports optional label_smoothing.
    """
    def __init__(self, class_counts: torch.Tensor, label_smoothing: float = 0.0):
        super().__init__()
        # store as plain tensor; NOT parameter/buffer (avoid state_dict issues)
        self.class_counts = class_counts.float().clamp_min(1)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, C), targets: (B,)
        counts = self.class_counts.to(logits.device)
        log_den = torch.logsumexp(logits + counts.log(), dim=1)  # (B,)
        num = logits.gather(1, targets.view(-1, 1)).squeeze(1)    # z_y
        log_prob = num - log_den                                  # log p_bal(y|x)
        nll = -log_prob

        if self.label_smoothing > 0:
            # smooth with uniform over classes
            C = logits.size(1)
            with torch.no_grad():
                one_hot = F.one_hot(targets, num_classes=C).float()
                smoothed = (1 - self.label_smoothing) * one_hot + self.label_smoothing / C
            # compute balanced softmax probabilities for all classes
            # log p_j = z_j - logsumexp(z + log n)
            log_probs_all = logits - log_den.unsqueeze(1) + 0.0  # reused denominator
            # KL from smoothed targets to log_probs_all:
            nll = -(smoothed * log_probs_all).sum(dim=1)

        return nll.mean()


class LDAMLoss(nn.Module):
    """
    Label-Distribution Aware Margin loss for multi-class:
      subtract a class-dependent margin from the true-class logit before CE.
    """
    def __init__(self, class_counts, power=0.25, max_m=0.5, scale=30.0, weight=None):
        super().__init__()
        self.class_counts = class_counts.float().clamp_min(1)
        m = (1.0 / (self.class_counts ** power))
        m = m * (max_m / m.max())  # scale to max_m
        self.margins = m  # plain tensor (no buffer)
        self.scale = float(scale)
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, targets):
        # subtract margin from the target logit
        margins = self.margins.to(logits.device)
        # build a margin vector per sample
        m_y = margins.gather(0, targets)
        logits_adj = logits.clone()
        logits_adj[torch.arange(logits.size(0)), targets] -= m_y
        return self.ce(self.scale * logits_adj, targets)


def modules_from_config(modules: list[tuple[str, dict]]):
    modules_list = []
    for module in modules:
        module_type = list(module)[0]
        config = module[module_type]
        if module_type == "linear":
            module = Linear(**config)
        elif module_type == "conv1d":
            module = Conv1d(**config)
        elif module_type == "softsign":
            module = Softsign()
        elif module_type == "relu":
            module = ReLU()
        elif module_type == "elu":
            module = ELU()
        elif module_type == "sigmoid":
            module = Sigmoid()
        elif module_type == "gelu":
            module = GELU()
        elif module_type == "dropout":
            module = nn.Dropout(**config)
        elif module_type == "dropout1d":
            module = nn.Dropout1d(**config)
        elif module_type == "gru":
            module = GRU(**config)
        elif module_type == "flatten":
            module = nn.Flatten()
        elif module_type == "batch_norm1d":
            module = BatchNorm1d(**config)
        elif module_type == "permute":
            module = Permute(**config)
        elif module_type == "resnet_block":
            module = ResnetBlock(**config)
        elif module_type == "layer_norm":
            module = LayerNorm(**config)
        elif module_type == "average_groups":
            module = AverageGroups(**config)
        elif module_type == "select_channels":
            module = SelectChannels(**config)
        elif module_type == "lstm_block":
            module = LSTMBlock(**config)
        elif module_type == "flatten_time":
            module = FlattenTime()
        elif module_type == "dyslexnet":
            module = DyslexNetTransformer(**config)
        elif module_type == "bertspeech":
            module = BertSpeech(**config)
        elif module_type == "conformer":
            module = ConformerSpeech(**config)
        elif module_type == "conformer_seq2seq":
            module = ConformerSeq2Seq(**config)
        elif module_type == "neural_ensemble":
            module = NeuralEnsemble(**config)
        elif module_type == "megnet":
            module = MEGNet(**config)
        else:
            raise ValueError(f"Unsupported module_type: {module_type}")
        modules_list.append(module)
    return modules_list


def optimizer_from_config(parameters, config):
    if (config["name"] == "adam"):
        optimizer = Adam(parameters, **
                         config["config"])
    elif (config["name"] == "adamw"):
        optimizer = AdamW(parameters, **
                          config["config"])
    elif (config["name"] == "sgd"):
        optimizer = SGD(parameters, **
                        config["config"])
    else:
        raise ValueError(f"Unsupported optimizer: "
                         + f"{config['name']}")

    if ("scheduler" in config):
        scheduler_interval = config.get("scheduler_interval", "step")

        if (config["scheduler"] == "linear"):
            scheduler = LinearLR(optimizer,
                                 **config["scheduler_config"])
        elif (config["scheduler"] == "step"):
            scheduler = StepLR(optimizer,
                               **config["scheduler_config"])
        elif (config["scheduler"] == "cosine"):
            scheduler = CosineAnnealingLR(optimizer,
                                          **config["scheduler_config"])
        elif (config["scheduler"] == "cosine_warm"):
            scheduler = CosineAnnealingWarmRestarts(optimizer,
                                                    **config["scheduler_config"])
        else:
            raise ValueError(f"Unsupported scheduler: ",
                             config["scheduler"])

        sched_dict = {
            "scheduler": scheduler,
            "interval": scheduler_interval,
            "frequency": 1,
            "name": "lr",
        }
        return {
            "optimizer": optimizer,
            "lr_scheduler": sched_dict
        }

    return optimizer


def loss_fn_from_config(loss_config):
    name = loss_config["name"]
    cfg = dict(loss_config.get("config", {}) or {})  # copy so we can edit safely

    def _to_tensor(x, dtype=torch.float32):
        if torch.is_tensor(x):
            return x
        if isinstance(x, (list, tuple)):
            return torch.tensor(x, dtype=dtype)
        return x

    if name == "cross_entropy":
        # rehydrate weight if it was saved as list
        if "weight" in cfg:
            cfg["weight"] = _to_tensor(cfg["weight"])
        cfg.pop("class_counts", None)
        cfg.pop("cb_beta", None)
        cfg.pop("alpha", None)
        cfg.pop("log_k", None)
        return nn.CrossEntropyLoss(**cfg)

    elif name == "bce_with_smoothing":
        smoothing  = float(cfg.pop("smoothing", .0))
        pos_weight = _to_tensor(cfg.pop("pos_weight", 1.0))  # handles list or scalar
        cfg.pop("class_counts", None)
        cfg.pop("cb_beta", None)
        cfg.pop("alpha", None)
        cfg.pop("log_k", None)
        return BCEWithLogitsLossWithSmoothing(smoothing, pos_weight, **cfg)

    elif name == "balanced_softmax":
        counts = _to_tensor(cfg.pop("class_counts", None))
        if counts is None:
            raise ValueError("balanced_softmax requires 'class_counts' in loss.config")
        ls = float(cfg.pop("label_smoothing", 0.0))
        return BalancedSoftmaxCE(counts, label_smoothing=ls)

    elif name == "ldam":
        counts = _to_tensor(cfg.pop("class_counts", None))
        if counts is None:
            raise ValueError("ldam requires 'class_counts' in loss.config")
        power  = float(cfg.pop("power", 0.25))
        max_m  = float(cfg.pop("max_margin", 0.5))
        scale  = float(cfg.pop("scale", 30.0))
        weight = _to_tensor(cfg.pop("weight", None))
        return LDAMLoss(counts, power=power, max_m=max_m, scale=scale, weight=weight)

    elif name == "focal":
        gamma     = float(cfg.pop("gamma", 2.0))
        reduction = cfg.pop("reduction", "mean")
        alpha     = _to_tensor(cfg.pop("alpha", cfg.pop("weight", None)))
        return FocalLoss(gamma=gamma, alpha=alpha, reduction=reduction)

    else:
        raise ValueError(f"Unsupported loss: {name}")


class ResnetBlock(nn.Module):
    def __init__(self, model_config: list[tuple[str, dict]]):
        super().__init__()
        self.model_config = model_config
        self.module_list = nn.ModuleList()
        self.module_list.extend(modules_from_config(model_config))

    def forward(self, x):
        x_residual = x
        for module in self.module_list:
            x = module(x)
        return x + x_residual
