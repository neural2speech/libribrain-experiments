import torch
from torch.nn import Conv1d, ELU
from torch.nn import Softsign, GRU, Linear, ReLU, Sigmoid, GELU, BatchNorm1d
from torch import nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import LinearLR, StepLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from libribrain_experiments.models.util_layers import Permute, LSTMBlock, SelectChannels
from libribrain_experiments.models.util_losses import BCEWithLogitsLossWithSmoothing
from torch.nn import LayerNorm
from libribrain_experiments.models.average_groups import AverageGroups
from libribrain_experiments.models.dyslexnet import DyslexNetTransformer
from libribrain_experiments.models.bertspeech import BertSpeech
from libribrain_experiments.models.conformer import ConformerSpeech
from libribrain_experiments.models.conformer_seq2seq import ConformerSeq2Seq


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
        elif module_type == "dyslexnet":
            module = DyslexNetTransformer(**config)
        elif module_type == "bertspeech":
            module = BertSpeech(**config)
        elif module_type == "conformer":
            module = ConformerSpeech(**config)
        elif module_type == "conformer_seq2seq":
            module = ConformerSeq2Seq(**config)
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
    if loss_config["name"] == "cross_entropy":
        if ("config" not in loss_config or loss_config["config"] is None):
            return nn.CrossEntropyLoss()
        return nn.CrossEntropyLoss(**loss_config["config"])
    elif loss_config["name"] == "bce_with_smoothing":
        cfg = loss_config.get("config", {})
        smoothing   = float(cfg.pop("smoothing", .0))
        pos_weight  = torch.tensor([cfg.pop("pos_weight", 1.0)])
        return BCEWithLogitsLossWithSmoothing(smoothing, pos_weight, **cfg)
    else:
        raise ValueError(f"Unsupported loss: {loss_config['name']}")


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
