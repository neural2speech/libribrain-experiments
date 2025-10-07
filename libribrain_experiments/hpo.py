import os
from argparse import ArgumentParser
import itertools
from libribrain_experiments.utils import (
    run_training,
    get_datasets_from_config,
    adapt_config_to_data,
    run_validation,
    log_results,
    get_holdout_dataset_from_config,
    write_holdout_predictions,
)
import yaml
import wandb
import pytorch_lightning as lightning
import numpy as np
import torch
import time
from libribrain_experiments.utils import get_label_counts
from lightning.pytorch.accelerators import find_usable_cuda_devices


def update_config_for_single_run(config: dict, run_config: list[tuple[tuple, list]]):
    """
        config: dict
        run_config: list of updates to the config. Each entry maps a keylist to a value e.g. (("optimizer", "config", "lr"), 0.01)
    """
    for key_list, value in run_config:
        current = config
        try:
            for key in key_list[:-1]:
                current = current[key]
            current[key_list[-1]] = value
        except KeyError:
            raise KeyError(
                f"Key list {key_list} not found in config. Config: {config}")
    return config


def runs_configs_from_search_space(search_space: dict[tuple, list]):
    """
        search_space: dict that maps key_list to the list of values to try for that key
        returns: list where each element describes all the hyperparameter updates for a single run
    """
    if (len(search_space) == 0):
        return []
    keys, values = zip(*search_space.items())
    result = []
    for v in itertools.product(*values):
        config = list(zip(keys, v))
        result.append(config)
    return result


def get_run(config, search_space, i):
    run_config = search_space[i]
    return update_config_for_single_run(config, run_config)


def load_search_space(path: str):
    try:
        with open(path, 'r') as f:
            search_space = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Search space file not found. Please provide a valid path")
    search_space = parse_search_space(search_space)
    return search_space


def parse_search_space(search_space: dict):
    result = {}
    for key, value in search_space.items():
        new_key = eval(key)
        result[new_key] = value
    return result


def main(args):
    start_time = time.time()
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            "Config file not found. Please provide a valid path")

    search_space = load_search_space(args.search_space)
    run_configs = runs_configs_from_search_space(search_space)
    if (args.run_index is None):
        args.run_index = np.random.randint(0, len(run_configs))
    config = get_run(config, run_configs, args.run_index)

    print("Running config: ", args.run_index)
    print("Config: ", run_configs[args.run_index])

    if (args.run_name is None):
        args.run_name = "hpo-run-" + str(args.run_index)
    else:
        args.run_name = args.run_name + "-hpo-" + str(args.run_index)
    config["general"]["run_name"] = args.run_name
    if (config["general"]["wandb"]):
        if (args.project_name is None):
            raise ValueError(
                "Please provide a project name for wandb logging")
        wandb.init(project=args.project_name, name=args.run_name)
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_f1_macro", summary="max")
        wandb.define_metric("val_bal_acc", summary="max")

    seed = config["general"]["seed"]
    lightning.seed_everything(seed)

    print("SEEDED EVERYTHING in ", time.time() - start_time, " seconds")
    start_time = time.time()

    train_dataset, val_dataset, test_dataset, labels = get_datasets_from_config(
        config["data"], seed
    )

    print("LOADED DATASETS in ", time.time() - start_time, " seconds")

    if ("train_fraction" in config["data"]["general"]):
        train_fraction = config["data"]["general"]["train_fraction"]
        train_size = int(len(train_dataset) * train_fraction)
        remaining_size = len(train_dataset) - train_size
        train_dataset, _ = torch.utils.data.random_split(
            train_dataset, [train_size, remaining_size])
    print("TRAIN SIZE: ", len(train_dataset))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, **config["data"]["dataloader"])
    val_loader = torch.utils.data.DataLoader(
        val_dataset, **config["data"]["dataloader"])
    adapt_config_to_data(config, train_loader, labels)

    start_time = time.time()
    print("ADAPTED CONFIG TO DATA in ", time.time() - start_time, " seconds")

    if "best_model_metrics" in config["general"]:
        best_model_metric = config["general"]["best_model_metrics"]
    else:
        best_model_metric = "val_bal_acc"

    if best_model_metric == "val_loss":
        best_model_metric_mode = "min"
    else:
        best_model_metric_mode = "max"

    trainer, best_module, module = run_training(
        train_loader, val_loader, config, len(labels), best_model_metric=best_model_metric, best_model_metric_mode=best_model_metric_mode)
    start_time = time.time()
    print("TRAINED MODEL in ", time.time() - start_time, " seconds")

    samples_per_class = get_label_counts(train_loader, len(labels))

    """result, y, preds, logits = run_validation(
        val_loader, module, labels, avg_evals=[5, 100], samples_per_class=samples_per_class)
    start_time = time.time()
    print("VALIDATED LAST MODEL in ", time.time() - start_time, " seconds")

    log_results(result, y, preds, logits,
                config["general"]["output_path"], "last-" + str(args.run_name), hpo_config=run_configs[args.run_index], trainer=trainer)
    start_time = time.time()
    print("LOGGED LAST RESULTS in ", time.time() - start_time, " seconds")"""
    eval_ckpt = config["general"].get("eval_checkpoint", "best")
    if   eval_ckpt == "last":
        model_for_eval = module
    elif eval_ckpt == "best":
        model_for_eval = best_module
    else:
        raise ValueError(
            f'Unknown evaluation_checkpoint "{eval_ckpt}". '
            'Use "best" or "last".')

    model_for_eval = model_for_eval.to(find_usable_cuda_devices()[0])
    del module, best_module

    result, y, preds, logits = run_validation(
        val_loader, model_for_eval, labels, avg_evals=[], samples_per_class=samples_per_class)
    start_time = time.time()
    print("VALIDATED MODEL in ", time.time() - start_time, " seconds")
    log_results(result, y, preds, logits,
                config["general"]["output_path"], f"val-{eval_ckpt}-" + str(args.run_name))
    start_time = time.time()
    print(f"LOGGED {eval_ckpt.upper()} RESULTS in ", time.time() - start_time, " seconds")

    if test_dataset is not None:
        print("Validating on test set")
        test_loader = torch.utils.data.DataLoader(
            test_dataset, **config["data"]["dataloader"])
        result, y, preds, logits = run_validation(
            test_loader, model_for_eval, labels, samples_per_class=samples_per_class)
        start_time = time.time()
        print("VALIDATED MODEL in ", time.time() - start_time, " seconds")
        log_results(result, y, preds, logits,
                    config["general"]["output_path"], f"test-{eval_ckpt}-" + str(args.run_name))
        start_time = time.time()
        print(f"LOGGED {eval_ckpt.upper()} RESULTS in ", time.time() - start_time, " seconds")

    # Write Holdout CSV(s)
    holdout_ds = get_holdout_dataset_from_config(config["data"])
    if holdout_ds is not None:
        # default output filename under the run's output directory
        out_dir = os.path.join(config["general"]["output_path"], f"holdout-{eval_ckpt}-{args.run_name}")
        os.makedirs(out_dir, exist_ok=True)
        out_csv = os.path.join(out_dir, "submission_phoneme.csv")
        print(f"Generating holdout predictions -> {out_csv}")
        write_holdout_predictions(
            model_for_eval,
            holdout_ds,
            dataloader_cfg=config["data"].get("dataloader", {}),
            out_csv=out_csv,
            device=model_for_eval.device,
        )


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--run-index", type=int,
                        help="Index of the run to execute. If none, random run will be chosen")
    parser.add_argument("--search-space", type=str, required=True)
    parser.add_argument("--run-name", type=str)
    parser.add_argument("--project-name", type=str,
                        default="libribrain-experiments")
    args = parser.parse_args()
    main(args)
