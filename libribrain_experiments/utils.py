import random
from pnpl.datasets import LibriBrainPhoneme, LibriBrainSpeech
from torch.utils.data import DataLoader, ConcatDataset
from pnpl.datasets.grouped_dataset import GroupedDataset
import json
import os
import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from torchmetrics import Accuracy, F1Score, Recall
from torchmetrics.classification import MulticlassAUROC, BinaryAUROC
from torchmetrics import JaccardIndex
from libribrain_experiments.models.configurable_modules.classification_module import ClassificationModule
import numpy as np
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
import json
import numpy as np


# These are the sensors we identified as being particularly useful
SENSORS_SPEECH_MASK = [18, 20, 22, 23, 45, 120, 138, 140, 142, 143, 145,
                      146, 147, 149, 175, 176, 177, 179, 180, 198, 271, 272, 275]


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


DATASETS = {
    "libribrain_phoneme": LibriBrainPhoneme,
    "libribrain_speech": LibriBrainSpeech,
    "libribrain_speech_simplified": LibriBrainSpeechSimplified,
    "libribrain_speech_filtered": lambda **kw: FilteredDataset(LibriBrainSpeech(**kw)),
}


def check_labels(list_of_labels):
    reference_labels = list_of_labels[0]
    for labels in list_of_labels[1:]:
        if (labels != reference_labels):
            raise ValueError(
                f"Datasets have different labels: {labels} and {reference_labels}")


def apply_dataset_wrappers_from_data_config(dataset, data_config):
    # applies dataset wrappers from data config
    if ("averaged_samples" in data_config["general"] and "grouped_samples" in data_config["general"]):
        raise ValueError(
            "Only one grouping type can be used at a time. Please change data config")
    if ("averaged_samples" in data_config["general"] and data_config["general"]["averaged_samples"] > 1):
        dataset = GroupedDataset(
            dataset, grouped_samples=data_config["general"]["averaged_samples"], average_grouped_samples=True)
    if ("grouped_samples" in data_config["general"] and data_config["general"]["grouped_samples"] > 1):
        dataset = GroupedDataset(
            dataset, grouped_samples=data_config["general"]["grouped_samples"], average_grouped_samples=False, drop_remaining=True)
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
        dataset = DATASETS[name](**config)
        partition_datasets.append(dataset)
        partition_dataset_labels.append(dataset.labels_sorted)
    # ensure all datasets have the same set of labels
    check_labels(partition_dataset_labels)
    partition_dataset = ConcatDataset(partition_datasets)
    return partition_dataset


def get_datasets_from_config(data_config):
    datasets_config = data_config["datasets"]

    if "train" in datasets_config:
        train_dataset = get_dataset_partition_from_config(
            datasets_config["train"])
        train_channel_means = train_dataset.datasets[0].channel_means
        train_channel_stds = train_dataset.datasets[0].channel_stds
        train_labels_sorted = train_dataset.datasets[0].labels_sorted
        train_dataset = apply_dataset_wrappers_from_data_config(
            train_dataset, data_config)
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
            val_dataset, data_config)
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
            test_dataset, data_config)
    else:
        test_dataset = None
    return train_dataset, val_dataset, test_dataset, train_labels_sorted


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
    label_counts = torch.zeros(n_classes)
    for batch in train_loader:
        y = batch[1]
        label_counts += torch.bincount(y, minlength=n_classes)
    return label_counts


def get_label_distribution(train_loader, n_classes):
    label_counts = get_label_counts(train_loader, n_classes)
    label_distribution = label_counts / label_counts.sum()
    return label_distribution


def run_training(train_loader, val_loader, config, n_classes, best_model_metric="val_f1_macro", module=None, best_model_metric_mode="max"):
    if module is None:
        module = ClassificationModule(
            model_config=config["model"], n_classes=n_classes,
            optimizer_config=config["optimizer"], loss_config=config["loss"],
            single_logit=config["general"].get("single_logit", False)
        )

    logger = False
    if (config["general"]["wandb"]):
        logger = WandbLogger()
    elif ("tensorboard_logger" in config["general"] and config["general"]["tensorboard_logger"]):
        logger = TensorBoardLogger(
            save_dir=config["general"]["checkpoint_path"])

    callbacks = []
    if ("early_stopping" in config["trainer"]):
        es_conf = config["trainer"].pop("early_stopping")
        callbacks.append(EarlyStopping(**es_conf))
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

    trainer_config = config["trainer"]
    trainer = Trainer(
        logger=logger,
        accelerator="auto",
        log_every_n_steps=1,
        callbacks=callbacks,
        **trainer_config
    )

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
            x = x.to(module.device)
            y = y.to(module.device)
            outputs = module(x)
            # single–logit -> 2-column soft-max
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


def adapt_config_to_data(config, train_loader, labels):
    n_classes = len(labels)
    class_weights = None
    if ("loss" in config and "config" in config["loss"] and "weight" in config["loss"]["config"]):
        if (config["loss"]["config"]["weight"] == "auto"):
            if (class_weights is None):
                class_weights = get_label_distribution(train_loader, n_classes)
            inverse_class_freq = 1 / class_weights
            config["loss"]["config"]["weight"] = inverse_class_freq

    if "grouped_samples" in config["data"]["general"]:
        for layer in config["model"]:
            layer_name = list(layer.keys())[0]
            layer_dict = layer[layer_name]
            if (layer_dict is None):
                continue
            if ("n_groups" in layer_dict and layer_dict["n_groups"] == "auto"):
                layer_dict["n_groups"] = config["data"]["general"]["grouped_samples"]
