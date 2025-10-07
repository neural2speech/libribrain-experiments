import csv
import random
from pnpl.datasets import LibriBrainPhoneme, LibriBrainSpeech
from pnpl.datasets import LibriBrainCompetitionHoldout
from libribrain_experiments.datasets.prediction_smoother import PredictionSmootherDataset
from torch.utils.data import DataLoader, ConcatDataset, Subset
from pnpl.datasets.grouped_dataset import GroupedDataset
import json
import os
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy, F1Score, Recall
from torchmetrics.classification import MulticlassAUROC, BinaryAUROC
from torchmetrics import JaccardIndex
from libribrain_experiments.augment import MEGAugment, DynamicGroupedDataset
from libribrain_experiments.augment import DynamicGroupedDatasetPerClass
from libribrain_experiments.models.configurable_modules.classification_module import ClassificationModule
from libribrain_experiments.models.configurable_modules.sequence_classification_module \
    import SequenceClassificationModule
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import json
import numpy as np




# These are the sensors we identified as being particularly useful
SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
                      146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]


class LibriBrainSpeechWithLabels(LibriBrainSpeech):
    """LibriBrainSpeech + .labels_sorted = [0, 1] so the old
    pipeline keeps working untouched."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # the two classes are always 0 = silence, 1 = speech
        self.labels_sorted = [0, 1]

class LibriBrainSpeechSimplified(torch.utils.data.Dataset):
    """
    Parameters:
        dataset: LibriBrain dataset.
        limit_samples (int, optional): If provided, limits the length of the dataset to this
                          number of samples.
        speech_silence_only (bool, optional): If True, only includes segments that are either
                          purely speech or purely silence (with additional balancing).
        apply_sensors_speech_mask (bool, optional): If True, applies a fixed sensor mask to the sensor
                          data in each sample.
    """

    def __init__(self, **kwargs):
        self.dataset = LibriBrainSpeech(**kwargs)
        self.labels_sorted = [0, 1]
        self.channel_means = self.dataset.channel_means
        self.channel_stds = self.dataset.channel_stds

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # Map index to the original dataset using balanced indices
        sample = list(self.dataset[index])

        label_middle_index = sample[1].shape[0] // 2
        sample[1] = sample[1][label_middle_index]
        return sample



class FilteredDataset(torch.utils.data.Dataset):
    """
    Parameters:
        dataset: LibriBrain dataset.
        limit_samples (int, optional): If provided, limits the length of the dataset to this
                          number of samples.
        speech_silence_only (bool, optional): If True, only includes segments that are either
                          purely speech or purely silence (with additional balancing).
        apply_sensors_speech_mask (bool, optional): If True, applies a fixed sensor mask to the sensor
                          data in each sample.
    """

    def __init__(self,
                 dataset,
                 limit_samples=None,
                 apply_sensors_speech_mask=True):
        super().__init__()

        self.dataset = dataset
        self.limit_samples = limit_samples
        self.apply_sensors_speech_mask = apply_sensors_speech_mask

        # These are the sensors we identified:
        self.sensors_speech_mask = SENSORS_SPEECH_MASK

        # Shuffle the indices
        self.balanced_indices = list(range(len(dataset)))
        self.balanced_indices = random.sample(self.balanced_indices, len(self.balanced_indices))

        # relay the mandatory bookkeeping attributes
        if hasattr(dataset, "labels_sorted"):
            self.labels_sorted = list(dataset.labels_sorted)
        else:  # LibriBrainSpeech has only 2 classes
            self.labels_sorted = [0, 1]
        self.channel_means = getattr(dataset, "channel_means", None)
        self.channel_stds  = getattr(dataset, "channel_stds",  None)

    def __len__(self):
        """Returns the number of samples in the filtered dataset."""
        if self.limit_samples is not None:
            return self.limit_samples
        return len(self.balanced_indices)

    def __getitem__(self, index):
        # Map index to the original dataset using balanced indices
        original_idx = self.balanced_indices[index]
        if self.apply_sensors_speech_mask:
            sensors = self.dataset[original_idx][0][self.sensors_speech_mask]
        else:
            sensors = self.dataset[original_idx][0][:]
        label_from_the_middle_idx = self.dataset[original_idx][1].shape[0] // 2
        return sensors, self.dataset[original_idx][1][label_from_the_middle_idx]


class MEGAugmentDataset(torch.utils.data.Dataset):
    """
    Thin wrapper that runs MEGAugment per sample.

    - Keeps labels / attrs intact
    - Should be applied to *train* split only
    """
    def __init__(self, base_dataset, augment_cfg: dict):
        super().__init__()
        self.base = base_dataset
        self.aug = MEGAugment(**augment_cfg)

        # Relay bookkeeping attributes
        for attr in ("labels_sorted", "channel_means", "channel_stds"):
            if hasattr(base_dataset, attr):
                setattr(self, attr, getattr(base_dataset, attr))

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        x = self.aug(x)
        return x, y


DATASETS = {
    "libribrain_phoneme": LibriBrainPhoneme,
    "libribrain_speech": LibriBrainSpeechWithLabels,
    "libribrain_speech_simplified": LibriBrainSpeechSimplified,
    "libribrain_speech_filtered": lambda **kw: FilteredDataset(LibriBrainSpeech(**kw)),
    "prediction_smoother": PredictionSmootherDataset,
}


class Unfreeze(Callback):
    def __init__(self, at_epoch=3):
        self.at_epoch = at_epoch

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch == self.at_epoch:
            if hasattr(pl_module, "modules_list"):
                for m in pl_module.modules_list:
                    for _name, p in m.named_parameters(recurse=True):
                        p.requires_grad = True
            else:
                # Fallback: unfreeze whole module
                for p in pl_module.parameters():
                    p.requires_grad = True

            # (optional) small log
            # n_trainable = sum(p.requires_grad for p in pl_module.parameters())
            # pl_module.print(f"[Unfreeze] epoch {self.at_epoch}: set requires_grad=True for all modules_list params "
            #                 f"(now trainable={n_trainable}).")
            print(f"[epoch {self.at_epoch}] encoder unfrozen")


class RegroupCallback(Callback):
    """Call dataset.reshuffle() at the start of every training epoch."""
    def __init__(self, grouped_ds: DynamicGroupedDataset):
        self.grouped_ds = grouped_ds

    def on_train_epoch_start(self, trainer, pl_module):
        self.grouped_ds.reshuffle()


def check_labels(list_of_labels):
    reference_labels = list_of_labels[0]
    for labels in list_of_labels[1:]:
        if (labels != reference_labels):
            raise ValueError(
                f"Datasets have different labels: {labels} and {reference_labels}")


def _multi_groupings(
    base_ds,
    num_sets: int,
    grouped_samples: int,
    *,
    average: bool,
    drop_remaining: bool,
) -> torch.utils.data.Dataset:
    """
    Build `num_sets` independent GroupedDataset views of `base_ds` with shuffle=True
    and concatenate them. If num_sets == 1, returns a single GroupedDataset with
    shuffle=False (keeps old behavior).
    """
    if num_sets <= 1:
        return GroupedDataset(
            base_ds,
            grouped_samples=grouped_samples,
            average_grouped_samples=average,
            drop_remaining=drop_remaining,
            shuffle=False,
        )
    parts = []
    for _ in range(num_sets):
        parts.append(
            GroupedDataset(
                base_ds,
                grouped_samples=grouped_samples,
                average_grouped_samples=average,
                drop_remaining=drop_remaining,
                shuffle=True,   # different random grouping each time
            )
        )
    return ConcatDataset(parts)


def _multi_groupings_var(
    base_ds,
    group_sizes: list[int],
    *,
    average: bool,
    drop_remaining: bool,
) -> torch.utils.data.Dataset:
    """
    Build one GroupedDataset per size in `group_sizes` (e.g., [100,100,50,20]),
    each with shuffle=True, and concatenate them. Duplicates are allowed.
    """
    parts = []
    for k in group_sizes:
        k = int(k)
        if k < 1:
            continue  # ignore invalid entries
        parts.append(
            GroupedDataset(
                base_ds,
                grouped_samples=k,
                average_grouped_samples=average,
                drop_remaining=drop_remaining,
                shuffle=True,
            )
        )
    if not parts:
        return base_ds
    return parts[0] if len(parts) == 1 else ConcatDataset(parts)


def apply_dataset_wrappers_from_data_config(dataset, data_config, split: str):
    """
    Apply grouping wrappers based on data_config.
    - For 'train': allow the dynamic wrapper (if requested).
    - For 'val'/'test': allow static grouping, with optional multi-grouping augmentation.
    """
    g = data_config.get("general", {})

    # Per-split group-augment factor (default 1 = no augmentation)
    group_aug_default = int(g.get("group_augment", 1) or 1)
    group_aug_split = int(g.get(f"group_augment_{split}", group_aug_default) or 1)

    # Drop policy (applies to both averaged/grouped static wrappers)
    avg_drop = bool(g.get("averaged_drop_remaining", True))

    # per-class dynamic sizes (train only)
    per_class_cfg = g.get("dynamic_averaged_samples_per_class", None)
    if split == "train" and per_class_cfg:
        # Accept either class *names* or numeric ids as keys
        # Also accept an optional "default" key to override the global default size(s)
        default_sizes = per_class_cfg.get("default", g.get("dynamic_averaged_samples", 100))

        # Build map: class-id -> sizes
        # If keys are strings (phoneme names), map via dataset.labels_sorted
        label_list = getattr(dataset, "datasets", [dataset])[0].labels_sorted \
                     if hasattr(dataset, "datasets") else getattr(dataset, "labels_sorted", None)
        if label_list is None:
            raise ValueError("Per-class grouping requires dataset.labels_sorted to map names to ids.")

        def key_to_id(k):
            # int or str id
            if isinstance(k, int):
                return k
            try:
                return int(k)
            except Exception:
                pass
            # name -> id
            if k not in label_list:
                raise ValueError(f"Unknown class/phoneme '{k}'. Available: {label_list}")
            return label_list.index(k)

        per_class_sizes = {}
        for k, v in per_class_cfg.items():
            if k == "default":
                continue
            cid = key_to_id(k)
            per_class_sizes[cid] = v

        print(f"Debug: Dynamic per-class grouped dataset ({split}), "
              f"default={default_sizes}, overrides={list(per_class_sizes.keys())}")
        return DynamicGroupedDatasetPerClass(
            dataset,
            per_class_sizes=per_class_sizes,
            default_sizes=default_sizes,
            average=True,
            drop_remaining=avg_drop,
        )

    # Dynamic averaging for train (reshuffle every epoch) takes precedence
    dyn_any = g.get("dynamic_averaged_samples", 0)

    if split == "train" and isinstance(dyn_any, (list, tuple)):
        sizes = [int(k) for k in dyn_any if int(k) >= 1]
        if sizes:
            print(f"Debug message: dynamic grouped dataset ({split}) with sizes={sizes}")
            return DynamicGroupedDataset(dataset, grouped_samples=sizes,
                                         average=True, drop_remaining=avg_drop)
    elif split == "train":
        dyn_k = int(dyn_any or 0)
        if dyn_k > 1:
            print(f"Debug message: dynamic grouped dataset ({split}) with K={dyn_k}")
            return DynamicGroupedDataset(dataset, grouped_samples=dyn_k,
                                         average=True, drop_remaining=avg_drop)

    # Static averaged grouping  (supports int OR list)
    avg_any = g.get("averaged_samples", 0)
    avg_cfg = g.get(f"averaged_samples_{split}", avg_any)

    if isinstance(avg_cfg, (list, tuple)):
        sizes = [int(k) for k in avg_cfg if int(k) >= 1]
        if sizes:
            ds = _multi_groupings_var(
                dataset,
                group_sizes=sizes,
                average=True,
                drop_remaining=avg_drop,
            )
            print(f"[{split}] static averaged grouping: sizes={sizes} (concat).")
            return ds
    else:
        avg_k = int(avg_cfg or 0)
        if avg_k > 1:
            ds = _multi_groupings(
                dataset,
                num_sets=group_aug_split,
                grouped_samples=avg_k,
                average=True,
                drop_remaining=avg_drop,
            )
            if group_aug_split > 1:
                print(f"[{split}] static averaged grouping: {avg_k}x, "
                      f"augmented via {group_aug_split} independent groupings (concat).")
            return ds

    # Static concatenation (non-averaged) grouping  (supports int OR list)
    grp_any = g.get("grouped_samples", 0)
    grp_cfg = g.get(f"grouped_samples_{split}", grp_any)

    if isinstance(grp_cfg, (list, tuple)):
        sizes = [int(k) for k in grp_cfg if int(k) >= 1]
        if sizes:
            ds = _multi_groupings_var(
                dataset,
                group_sizes=sizes,
                average=False,
                drop_remaining=avg_drop,
            )
            print(f"[{split}] static concatenated grouping: sizes={sizes} (concat).")
            return ds
    else:
        grp_k = int(grp_cfg or 0)
        if grp_k > 1:
            ds = _multi_groupings(
                dataset,
                num_sets=group_aug_split,
                grouped_samples=grp_k,
                average=False,
                drop_remaining=avg_drop,
            )
            if group_aug_split > 1:
                print(f"[{split}] static concatenated grouping: {grp_k}x, "
                      f"augmented via {group_aug_split} independent groupings (concat).")
            return ds

    # No grouping requested
    return dataset


def get_dataset_partition_from_config(partition_config, channel_means=None, channel_stds=None):
    # loads datasets from config
    # returns concatenated dataset
    partition_dataset_names = [list(ds.keys())[0] for ds in partition_config]
    partition_dataset_configs = [list(ds.values())[0]
                                 for ds in partition_config]

    for config in partition_dataset_configs:
        # We can disable this to behave like the notebook if needed
        inherit_stats = config.pop("use_train_stats", True)

        # for simplicity we standardize using the first training dataset
        if (config.get("standardize", True)) and inherit_stats:
            config['channel_means'] = channel_means.tolist(
            ) if channel_means is not None else None
            config['channel_stds'] = channel_stds.tolist(
            ) if channel_stds is not None else None

    partition_datasets = []
    partition_dataset_labels = []
    for name, config in zip(partition_dataset_names, partition_dataset_configs):
        if (name not in DATASETS):
            raise ValueError(
                f"Dataset {name} not supported. Please change data config")
        # Extract the MEG Agument config
        mega_cfg  = config.pop("meg_augment", None)

        dataset = DATASETS[name](**config)

        if mega_cfg:
            if isinstance(mega_cfg, bool):
                mega_cfg = {}  # all defaults
            dataset = MEGAugmentDataset(dataset, mega_cfg)
        partition_datasets.append(dataset)
        partition_dataset_labels.append(dataset.labels_sorted)
    # ensure all datasets have the same set of labels
    check_labels(partition_dataset_labels)
    partition_dataset = ConcatDataset(partition_datasets)
    return partition_dataset


def limit_dataset(ds, max_examples: int, seed: int = 42):
    """Return a Subset with at most max_examples items, sampled once at init.
    """
    max_examples = int(max_examples)
    if max_examples <= 0 or max_examples >= len(ds):
        return ds
    g = torch.Generator()
    g.manual_seed(int(seed))
    idx = torch.randperm(len(ds), generator=g)[:max_examples].tolist()
    return Subset(ds, idx)


def get_datasets_from_config(data_config, seed=42):
    datasets_config = data_config["datasets"]

    if "train" in datasets_config:
        train_dataset = get_dataset_partition_from_config(
            datasets_config["train"])
        train_channel_means = train_dataset.datasets[0].channel_means
        train_channel_stds = train_dataset.datasets[0].channel_stds
        train_labels_sorted = train_dataset.datasets[0].labels_sorted
        train_dataset = apply_dataset_wrappers_from_data_config(
            train_dataset, data_config, "train")
        # Optional: cap the number of training examples
        train_limit = data_config.get("general", {}).get("train_limit", None)
        if train_limit is not None:
            seed = data_config.get("general", {}).get("train_limit_seed", seed)
            train_dataset = limit_dataset(train_dataset, train_limit, seed=seed)
    else:
        train_dataset = None
        train_labels_sorted = None
        train_channel_means = None
        train_channel_stds = None
    if "val" in datasets_config:
        val_dataset = get_dataset_partition_from_config(
            datasets_config["val"], train_channel_means, train_channel_stds)
        if train_labels_sorted is not None:
            check_labels(
                [train_labels_sorted, val_dataset.datasets[0].labels_sorted])
        val_dataset = apply_dataset_wrappers_from_data_config(
            val_dataset, data_config, "val")
    else:
        val_dataset = None
    if train_labels_sorted is None:  # HACKY FOR ARMENI COMPARISON
        train_labels_sorted = val_dataset.datasets[0].labels_sorted
    if "test" in datasets_config:
        test_dataset = get_dataset_partition_from_config(
            datasets_config["test"], train_channel_means, train_channel_stds)
        if train_labels_sorted is not None:
            check_labels(
                [train_labels_sorted, test_dataset.datasets[0].labels_sorted])
        test_dataset = apply_dataset_wrappers_from_data_config(
            test_dataset, data_config, "test")
    else:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset, train_labels_sorted


def collate_pad_maybe_unlabeled(batch):
    """
    Collate function that supports unlabeled holdout samples (x) OR labeled (x,y).
    Pads to the max time-length in the batch.
    Returns (xs, ys_or_None).
    """
    if isinstance(batch[0], torch.Tensor):  # holdout: only x
        xs_list = [b.clone() for b in batch]
        ys = None
    else:
        xs_list = [torch.as_tensor(b[0]).clone() for b in batch]
        ys = torch.stack([b[1] for b in batch])
    T_max = max(t.shape[-1] for t in xs_list)
    xs_pad = [torch.nn.functional.pad(t, (0, T_max - t.shape[-1])) for t in xs_list]
    xs = torch.stack(xs_pad, 0)  # (B, C, T_max)
    return xs, ys


def get_holdout_dataset_from_config(data_config):
    """
    Optional: read a 'holdout' entry under data.datasets and build the
    LibriBrainCompetitionHoldout dataset. We intentionally keep this
    separate from the train/val/test pipeline (no label checks, no grouping).

    Expected YAML shape:
      data:
        datasets:
          holdout:
            - libribrain_holdout:
                data_path: "<DATA_PATH>"
                tmin: 0.0
                tmax: 0.5
                standardize: false   # the holdout is already standardized
                channel_means: null  # leave None unless you really need it
                channel_stds:  null
                task: "phoneme"
        dataloader:
          batch_size: 256
          num_workers: 4
    """
    ds_cfg = data_config.get("datasets", {}).get("holdout")
    if not ds_cfg:
        return None

    # We support exactly one entry for holdout.
    if not isinstance(ds_cfg, (list, tuple)) or len(ds_cfg) != 1:
        raise ValueError("`data.datasets.holdout` must be a single-item list.")
    name = list(ds_cfg[0].keys())[0]
    cfg  = list(ds_cfg[0].values())[0] or {}

    if name not in ("libribrain_holdout", "competition_holdout"):
        raise ValueError(f"Unknown holdout dataset '{name}'. "
                         f"Use 'libribrain_holdout'.")

    # Sensible defaults (match your submission script)
    cfg.setdefault("standardize", False)
    cfg.setdefault("channel_means", None)
    cfg.setdefault("channel_stds",  None)
    cfg.setdefault("task", "phoneme")

    ds = LibriBrainCompetitionHoldout(**cfg)
    return ds


@torch.inference_mode()
def write_holdout_predictions(
    model: torch.nn.Module,
    holdout_ds,
    *,
    dataloader_cfg: dict,
    out_csv: str,
    device: torch.device | str | None = None,
):
    """
    Run the given (already-trained) model on `holdout_ds` and write a CSV.
    If the dataset exposes `generate_submission_in_csv`, we use that to match
    the expected competition file shape/order.
    """
    dev = device or (model.device if hasattr(model, "device") else "cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(dev)

    loader = DataLoader(
        holdout_ds,
        batch_size=dataloader_cfg.get("batch_size", 256),
        shuffle=False,
        num_workers=dataloader_cfg.get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_pad_maybe_unlabeled,
    )

    all_probs = []
    for xs, _ in loader:
        xs = xs.to(dev, non_blocking=True)
        logits = model(xs)                   # (B, C) or (B,1)
        if logits.dim() == 1:
            logits = logits.unsqueeze(1)
        probs = torch.softmax(logits, dim=1) # (B, C)
        all_probs.append(probs.cpu())

    probs_cat = torch.cat(all_probs, dim=0)  # (N, C) on CPU

    # Prefer the dataset's helper if available (keeps ordering/format right)
    if hasattr(holdout_ds, "generate_submission_in_csv"):
        tensors = [row.clone() for row in probs_cat]  # list[Tensor(C)]
        holdout_ds.generate_submission_in_csv(tensors, out_csv)
    else:
        os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            C = probs_cat.shape[1]
            w.writerow(["segment_idx"] + [f"class_{i}" for i in range(C)])
            for i, row in enumerate(probs_cat.numpy()):
                w.writerow([i, *row.tolist()])
    print(f"[holdout] Wrote {len(probs_cat):,} rows -> {out_csv}")


def log_results(result, y, preds, logits, output_path, run_name, hpo_config=None, trainer=None):
    if (hpo_config is not None):
        for conf in hpo_config:
            keys = [str(c) for c in conf[0]]
            key = "_".join(keys)
            value = conf[1]
            result[key] = value
    if (trainer is not None):
        result["train_loss"] = trainer.callback_metrics.get("train_loss")
    if (wandb.run is not None):
        wandb.log(result)
    result["targets"] = y
    result["preds"] = preds
    result["logits"] = logits
    del result["val_cm"]
    for key, value in result.items():
        if (isinstance(value, torch.Tensor)):
            result[key] = value.cpu().tolist()
        if (isinstance(value, np.ndarray)):
            result[key] = value.tolist()

    output_path = os.path.join(output_path, run_name)

    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "results.json"), "w") as f:
        json.dump(result, f)


def get_label_counts(train_loader, n_classes):
    """Return counts per class handling both (B,) and (B,T) targets."""
    label_counts = torch.zeros(n_classes, dtype=torch.long)
    for batch in train_loader:
        y = batch[1]
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)
        if y.dim() > 1:
            y = y.reshape(-1)
        y = y.to(torch.long)
        label_counts += torch.bincount(y, minlength=n_classes)
    return label_counts


def get_label_distribution(train_loader, n_classes):
    label_counts = get_label_counts(train_loader, n_classes)
    label_distribution = label_counts / label_counts.sum()
    return label_distribution


def run_training(train_loader, val_loader, config, n_classes, best_model_metric="val_f1_macro", module=None, best_model_metric_mode="max"):
    def _labels_are_sequences(loader) -> bool:
        """Peek one batch to see if targets are (B,) or (B,T)."""
        xb, yb = next(iter(loader))
        return yb.dim() == 2          # (B,T) -> sequence

    if module is None:
        ModuleCls = (
            SequenceClassificationModule
            if _labels_are_sequences(train_loader)
            else ClassificationModule
        )
        module = ModuleCls(
            model_config=config["model"], n_classes=n_classes,
            optimizer_config=config["optimizer"], loss_config=config["loss"],
            single_logit=config["general"].get("single_logit", False)
        )

    logger = False
    if (config["general"]["wandb"]):
        logger = WandbLogger()
        logger.watch(module, log="all", log_freq=100)
    elif ("tensorboard_logger" in config["general"] and config["general"]["tensorboard_logger"]):
        logger = TensorBoardLogger(
            save_dir=config["general"]["checkpoint_path"])

    callbacks = []
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    if ("early_stopping" in config["trainer"]):
        es_conf = config["trainer"].pop("early_stopping")
        callbacks.append(EarlyStopping(**es_conf))
    if ("unfreeze_epoch" in config["trainer"]):
        unfreeze_epoch = config["trainer"].pop("unfreeze_epoch")
        callbacks.append(Unfreeze(at_epoch=unfreeze_epoch))
    if (config["general"]["checkpoint_path"] is not None):
        os.makedirs(config["general"]["checkpoint_path"], exist_ok=True)
        checkpoint_callback = ModelCheckpoint(
            dirpath=config["general"]["checkpoint_path"],
            monitor=best_model_metric,  # Metric to monitor
            mode=best_model_metric_mode,          # Higher is better
            save_top_k=1,        # Save only the best checkpoint
            verbose=True,
            filename="best-" + best_model_metric +
            "-" + str(config["general"]["run_name"]) +
            "-{epoch:02d}-{val_f1_macro:.4f}",
            save_last=True
        )
        callbacks.append(checkpoint_callback)
    # Add callback for dynamic_averaged_samples shuffling at each epoch
    if isinstance(train_loader.dataset, DynamicGroupedDataset):
        callbacks.append(RegroupCallback(train_loader.dataset))

    trainer_config = config["trainer"]
    trainer = Trainer(
        logger=logger,
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=callbacks,
        **trainer_config
    )

    # Monitor model gradients and weights
    if (config["general"]["wandb"]):
        logger.experiment.watch(module, log_freq=100, log="all")

    trainer.fit(module, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    print("Debug message: loading: ", str(
        checkpoint_callback.best_model_path,))
    best_module = ClassificationModule.load_from_checkpoint(
        checkpoint_callback.best_model_path,
    )

    return trainer, best_module, module


def run_validation(val_loader, module, labels, avg_evals=None, samples_per_class=None):
    disp_labels = labels
    module.eval()
    all_preds = []
    all_logits = []
    all_targets = []
    all_probas = []
    single_logit = False
    with torch.no_grad():
        for batch in val_loader:
            x, y = batch[0], batch[1]
            # if torch.is_tensor(y) and y.dim() > 1:  # (B, T) -> (BxT)
            #     y = y.reshape(-1)
            x = x.to(module.device)
            y = y.to(module.device)
            outputs = module(x)
            # single-logit -> 2-column soft-max
            if outputs.dim() == 2 and outputs.size(1) == 1:  # (B,1)
                single_logit = True
                p = torch.sigmoid(outputs)                   # (B,1)
                outputs = torch.cat([1.0 - p, p], dim=1)     # (B,2)
            all_logits.extend(outputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds)
            all_targets.extend(y)
            all_probas.extend(torch.nn.functional.softmax(outputs, dim=1))
    # Compare with Naive Baseline
    all_targets = torch.stack(all_targets)
    all_preds = torch.stack(all_preds)
    all_logits = torch.stack(all_logits)
    all_probas = torch.stack(all_probas)
    # all_targets = torch.cat(all_targets)
    # all_preds = torch.cat(all_preds)
    # all_logits = torch.cat(all_logits)
    # all_probas = torch.cat(all_probas)

    if (samples_per_class is not None):
        bincount = samples_per_class.to(module.device)
    else:
        import warnings
        warnings.warn(
            "No samples per class provided, using bincount of val dataset")
        bincount = torch.bincount(all_targets).to(module.device)
    most_common_class = torch.argmax(bincount)
    naive_acc = bincount[most_common_class] / len(all_targets)

    num_classes = max(len(disp_labels), 2)
    acc = Accuracy(task="multiclass", average="micro",
                   num_classes=num_classes).to(module.device)
    bal_acc = Accuracy(task="multiclass", average="macro",
                       num_classes=num_classes).to(module.device)
    f1_macro = F1Score(task="multiclass", average="macro",
                       num_classes=num_classes).to(module.device)
    f1_micro = F1Score(task="multiclass", average="micro",
                       num_classes=num_classes).to(module.device)
    f1_weighted = F1Score(task="multiclass", average="weighted",
                          num_classes=num_classes).to(module.device)
    rocauc_macro = MulticlassAUROC(average="macro",
                                   num_classes=num_classes).to(module.device)
    rocauc_micro = MulticlassAUROC(average="weighted",
                                   num_classes=num_classes).to(module.device)
    if single_logit:  # binary -> BCE; expects (B,) logits / probs
        bce = torch.nn.BCEWithLogitsLoss()
        loss = bce(all_logits[:, 1], all_targets.float())  # use the positive logit
    else:  # multiclass -> CE
        loss = torch.nn.CrossEntropyLoss().to(module.device)(all_logits, all_targets)
    random_preds = torch.randint(
        0, len(disp_labels), (len(all_targets),), device=module.device)
    random_acc = acc(random_preds, all_targets)
    random_balanced_acc = bal_acc(random_preds, all_targets)
    random_f1_macro = f1_macro(
        random_preds, all_targets)
    random_f1_micro = f1_micro(
        random_preds, all_targets)
    random_f1_weighted = f1_weighted(
        random_preds, all_targets)
    naive_preds = torch.multinomial(
        bincount.float(), len(all_targets), replacement=True).to(module.device)
    naive_acc = acc(naive_preds, all_targets)
    naive_balanced_acc = bal_acc(
        naive_preds, all_targets)
    naive_f1_macro = f1_macro(
        naive_preds, all_targets)
    naive_f1_micro = f1_micro(
        naive_preds, all_targets)
    naive_f1_weighted = f1_weighted(
        naive_preds, all_targets)

    # calculate loss
    frequencies = bincount.float() / bincount.sum()
    naive_probas = [frequencies for _ in all_targets]
    naive_probas = torch.stack(naive_probas)
    naive_loss = torch.nn.NLLLoss()(torch.log(naive_probas), all_targets)

    naive_rocauc_macro = rocauc_macro(
        naive_probas, all_targets)
    naive_rocauc_micro = rocauc_micro(
        naive_probas, all_targets)

    acc_val = acc(all_preds, all_targets)
    bal_acc_val = bal_acc(all_preds, all_targets)
    f1_macro_val = f1_macro(all_preds, all_targets)
    f1_micro_val = f1_micro(all_preds, all_targets)
    rocauc_macro_val = rocauc_macro(all_probas, all_targets)
    rocauc_micro_val = rocauc_micro(all_probas, all_targets)
    f1_weighted_val = f1_weighted(all_preds, all_targets)

    result = {
        "val_cm": wandb.plot.confusion_matrix(
            probs=None,
            y_true=all_targets.cpu().numpy(),
            preds=all_preds.cpu().numpy(),
            class_names=disp_labels
        ),
        "val_naive_acc": naive_acc,
        "val_random_acc": random_acc,
        "val_random_bal_acc": random_balanced_acc,
        "val_random_f1_macro": random_f1_macro,
        "val_random_f1_micro": random_f1_micro,
        "val_random_f1_weighted": random_f1_weighted,
        "val_naive_acc": naive_acc,
        "val_naive_bal_acc": naive_balanced_acc,
        "val_naive_f1_macro": naive_f1_macro,
        "val_naive_f1_micro": naive_f1_micro,
        "val_naive_f1_weighted": naive_f1_weighted,
        "val_naive_loss": naive_loss,
        "val_acc": acc_val,
        "val_bal_acc": bal_acc_val,
        "val_f1_macro": f1_macro_val,
        "val_f1_micro": f1_micro_val,
        "val_f1_weighted": f1_weighted_val,
        "val_loss": loss,
        "val_rocauc_macro": rocauc_macro_val,
        "val_rocauc_micro": rocauc_micro_val,
        "val_naive_rocauc_macro": naive_rocauc_macro,
        "val_naive_rocauc_micro": naive_rocauc_micro,
    }
    if (len(disp_labels) == 2):
        jaccard_index = JaccardIndex(
            task="multiclass", num_classes=2).to(module.device)
        jaccard_index_val = jaccard_index(all_preds, all_targets)
        jaccard_index_naive = jaccard_index(naive_preds, all_targets)
        result["val_jaccard_index"] = jaccard_index_val
        result["val_naive_jaccard_index"] = jaccard_index_naive

    binary_acc = Accuracy(task="binary").to(module.device)
    binary_bal_acc = Recall(task="multiclass", num_classes=2,
                            average="macro").to(module.device)
    binary_f1 = F1Score(task="binary").to(module.device)
    binary_rocauc = BinaryAUROC().to(module.device)
    classes = all_targets.unique()
    for c in classes:
        class_probas = all_probas[:, c]
        class_preds = all_preds == c
        class_targets = all_targets == c
        class_acc = binary_acc(class_preds, class_targets)
        class_f1 = binary_f1(class_preds, class_targets)
        class_bal_acc = binary_bal_acc(class_preds, class_targets)
        class_random_preds = random_preds == c
        class_random_acc = binary_acc(class_random_preds, class_targets)
        class_random_f1 = binary_f1(class_random_preds, class_targets)
        class_naive_preds = naive_preds == c
        class_naive_acc = binary_acc(class_naive_preds, class_targets)
        class_naive_bal_acc = binary_bal_acc(class_naive_preds, class_targets)
        class_naive_f1 = binary_f1(class_naive_preds, class_targets)
        class_rocauc = binary_rocauc(class_probas, class_targets)
        result[f"val_class_{c}_acc"] = class_acc
        result[f"val_class_{c}_f1"] = class_f1
        result[f"val_class_{c}_random_acc"] = class_random_acc
        result[f"val_class_{c}_random_f1"] = class_random_f1
        result[f"val_class_{c}_naive_acc"] = class_naive_acc
        result[f"val_class_{c}_naive_bal_acc"] = class_naive_bal_acc
        result[f"val_class_{c}_naive_f1"] = class_naive_f1
        result[f"val_class_{c}_bal_acc"] = class_bal_acc
        result[f"val_class_{c}_rocauc"] = class_rocauc
    return result, all_targets.cpu().numpy(), all_preds.cpu().numpy(), all_logits.cpu().numpy()


# def adapt_config_to_data(config, train_loader, labels):
#     n_classes = len(labels)
#     class_weights = None
#     if ("loss" in config and "config" in config["loss"] and "weight" in config["loss"]["config"]):
#         if (config["loss"]["config"]["weight"] == "auto"):
#             if (class_weights is None):
#                 class_weights = get_label_distribution(train_loader, n_classes)
#             inverse_class_freq = 1 / class_weights
#             config["loss"]["config"]["weight"] = inverse_class_freq
# 
#     if "grouped_samples" in config["data"]["general"]:
#         for layer in config["model"]:
#             layer_name = list(layer.keys())[0]
#             layer_dict = layer[layer_name]
#             if (layer_dict is None):
#                 continue
#             if ("n_groups" in layer_dict and layer_dict["n_groups"] == "auto"):
#                 layer_dict["n_groups"] = config["data"]["general"]["grouped_samples"]


def _normalize_weights(w: torch.Tensor) -> torch.Tensor:
    w = w.float()
    return w / w.mean().clamp_min(1e-12)


def _effective_num_weights(counts: torch.Tensor, beta: float = 0.9999) -> torch.Tensor:
    counts = counts.float()
    beta = float(beta)
    eff = 1.0 - (beta ** counts)
    w = (1.0 - beta) / eff.clamp_min(1e-12)
    return _normalize_weights(w)


def _log_weights(counts: torch.Tensor, k: float = 1.0) -> torch.Tensor:
    """(optional) 1 / log(k + n_c) scheme"""
    counts = counts.float().clamp_min(1)
    w = 1.0 / torch.log(k + counts)
    return _normalize_weights(w)


def adapt_config_to_data(config, train_data_or_loader, labels):
    n_classes = len(labels)
    counts = get_label_counts(train_data_or_loader, n_classes).clamp_min(1)  # (C,)

    # expose counts so losses like BalancedSoftmax/LDAM can use them (as list for JSON safety)
    loss_cfg = config.get("loss", {}).get("config", {})
    loss_cfg["class_counts"] = counts.tolist()

    # Loss weights
    if ("loss" in config and "config" in config["loss"] and
        "weight" in config["loss"]["config"]):
        w_mode = config["loss"]["config"]["weight"]

        if isinstance(w_mode, str) and w_mode.startswith("auto"):
            if w_mode in ("auto", "auto_inv"):
                w = 1.0 / counts.float()
            elif w_mode in ("auto_inv_sqrt", "auto_sqrt"):
                w = 1.0 / counts.float().sqrt()
            elif w_mode == "auto_pow":
                alpha = float(config["loss"]["config"].get("alpha", 0.5))
                w = 1.0 / (counts.float() ** alpha)
            elif w_mode == "auto_cb":  # ENS
                beta = float(config["loss"]["config"].get("cb_beta", 0.9999))
                eff = 1.0 - (beta ** counts.float())
                w = (1.0 - beta) / eff.clamp_min(1e-12)
            elif w_mode == "auto_log":
                k = float(config["loss"]["config"].get("log_k", 1.0))
                w = 1.0 / torch.log(k + counts.float())
            else:
                raise ValueError(f"Unknown weight mode: {w_mode}")

            # normalize and clamp (same behavior as before)
            w = w / w.mean().clamp_min(1e-12)
            max_w = float(config["loss"]["config"].get("max_weight", 0.0))
            if max_w and max_w > 0:
                w = torch.clamp(w, max=max_w)

            # store as list to survive hparams saving
            config["loss"]["config"]["weight"] = w.tolist()

    # Auto n_groups kept as-is
    if "grouped_samples" in config["data"]["general"]:
        for layer in config["model"]:
            layer_name = list(layer.keys())[0]
            layer_dict = layer[layer_name]
            if layer_dict is None:
                continue
            if ("n_groups" in layer_dict and layer_dict["n_groups"] == "auto"):
                layer_dict["n_groups"] = config["data"]["general"]["grouped_samples"]
