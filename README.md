# LibriBrain Experiments: neural2speech Team

> Fork of the official **LibriBrain Competition 2025** starter kit
> Upstream: https://github.com/neural-processing-lab/libribrain-experiments

This fork contains the exact code and checkpoints for the **neural2speech**
team's submissions to both competition tracks:

- [**Speech Detection (Standard)**](https://neural-processing-lab.github.io/2025-libribrain-competition/leaderboard/speech_detection_standard/)
- [**Phoneme Classification (Standard)**](https://neural-processing-lab.github.io/2025-libribrain-competition/leaderboard/phoneme_classification_standard/)

All commands below reproduce our submitted runs (training, test evaluation, and holdout CSV generation).

---
## Installation

You can install the package in editable/development mode so that any local changes are immediately reflected:

```bash
pip install -e .
```

## Experiment Configuration

Configuration files for the phoneme decoding experiments detailed in our paper can be found in:

```
libribrain_experiments/phoneme/configs
```

**Important Configuration Notes:**

Before running the project, make sure to update the configuration files with the correct local paths:

- **`data_path`**: Specify the paths for your training, validation, and testing datasets.
- **`output_path`**: Set this to the directory where output results (e.g., logs, predictions) will be saved.
- **`checkpoint_path`**: Define the location where model checkpoints should be stored.

---

## Running an Experiment

Use the following command format to execute an experiment:

```bash
python libribrain_experiments/hpo.py \
    --config=libribrain_experiments/configs/phoneme/<config-name>/base-config.yaml \
    --search-space=libribrain_experiments/configs/phoneme/<config-name>/search-space.yaml \
    --run-name=<run-name> \
    --run-index=<run-id>
```

Replace `<config-name>`, `<run-name>`, and `<run-id>` with your own values—`<config-name>` selects which experiment folder under `libribrain_experiments/phoneme/configs`, `<run-name>` is the Weights & Biases run name, and `<run-id>` is the hyperparameter/seed configuration index.

---

## Speech Detection Model

Training the models:

```bash
python -m "libribrain_experiments.hpo" \
    --config="configs/speech/conformer/base-config-best-2025-07-28.yaml" \
    --search-space="configs/speech/conformer/search-space-S.yaml" \
    --run-name="conformer-S-best-2025-07-28" \
    --run-index="0"
```

Evaluating the model on the test set:

```bash
python -m "libribrain_experiments.make_submission" \
  --split "test" \
  --tmax 2.5 \
  --min_speech_len 100 \
\<CHECKPOINTS_PATH\>/final-speech-results/best-val_f1_macro-conformer-S-best-2025-07-28-hpo-0-epoch=16-val_f1_macro=0.8706.ckpt
```

Generating holdout submission file:

```bash
python -m "libribrain_experiments.make_submission" \
  --split "holdout" \
  --tmax 2.5 \
  --min_speech_len 100 \
  --output submission-speech.csv \
\<CHECKPOINTS_PATH\>/final-speech-results/best-val_f1_macro-conformer-S-best-2025-07-28-hpo-0-epoch=16-val_f1_macro=0.8706.ckpt
```

## Phoneme Classification Model

Training the models (seed 0 example):

```bash
python -m "libribrain_experiments.hpo" \
    --config="configs/phoneme/conformer/conformer-custom-2025-09-09-config.yaml" \
    --search-space="configs/phoneme/conformer/search-space.yaml" \
    --run-name="conformer-custom-2025-09-09" \
    --run-index="0"
```

Evaluating the models on the test set:

```bash
python -m "libribrain_experiments.make_submission_phoneme" \
  --split "test" --mimic_holdout_style \
  --ensemble "majority_vote" \
  -- \<CHECKPOINTS_PATH\>/final-results/best-val_f1_macro-conformer-custom-2025-09-09-*
```

Generating holdout submission file:

```bash
python -m "libribrain_experiments.make_submission_phoneme" \
  --split "holdout" \
  --ensemble "majority_vote" \
  --output "submission-phoneme.csv" \
  \<CHECKPOINTS_PATH\>/final-results/best-val_f1_macro-conformer-custom-2025-09-09-*
```

Here, we performed ensembling using 5 seeds.

## Trained Checkpoints

The checkpoints used to generate the scores are available here:

https://aholab.ehu.eus/~xzuazo/libribrain/
