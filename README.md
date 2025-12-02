# MEGConformer — Official Code for the LibriBrain 2025 Submission

[![arXiv - Paper](https://img.shields.io/badge/cs.CL-2512.01443-b31b1b?&logo=arxiv&logoColor=red)](https://arxiv.org/abs/2512.01443)
[![Slides](https://img.shields.io/badge/Slides-Drive-yellow.svg)](https://docs.google.com/presentation/d/1RvMWcotexnKuJF75gEl4Ze2JZVsiEOkuTJNHOjiCcKI/edit?usp=sharing)
[![Team](https://img.shields.io/badge/Team-%23neural2speech-green.svg)](https://www.isca-archive.org/iberspeech_2024/dezuazo24b_iberspeech.html)
[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/license/bsd-3-clause)
[![HiTZ](https://img.shields.io/badge/HiTZ-Basque%20Center%20for%20Language%20Technology-blueviolet)](http://www.hitz.eus/)

**#neural2speech Team — UPV/EHU, [HiTZ Center](https://www.hitz.eus/) & [Basque Center on Cognition, Brain and Language (BCBL)](https://www.bcbl.eu/en)**

> Fork of the official **LibriBrain Competition 2025** starter kit
> Upstream: https://github.com/neural-processing-lab/libribrain-experiments

This repository contains the exact code, configurations, and checkpoints used in our paper:

* [**MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification**](https://arxiv.org/abs/2512.01443)
(See citation [below](#citation))

It reproduces our Standard Track submissions for both LibriBrain 2025 tasks:

- [**Speech Detection (Standard)**](https://neural-processing-lab.github.io/2025-libribrain-competition/leaderboard/speech_detection_standard/)
- [**Phoneme Classification (Standard)**](https://neural-processing-lab.github.io/2025-libribrain-competition/leaderboard/phoneme_classification_standard/)

All commands below reproduce our submitted runs (training, test evaluation, and holdout CSV generation).

## Short Description

This work adapts a compact Conformer encoder to non-invasive MEG signals for two foundational tasks:

* **Speech Detection:** speech vs. silence
* **Phoneme Classification:** 39-way perceptual phoneme decoding

Key contributions include:

* A **unified [Conformer architecture](https://arxiv.org/abs/2005.08100)** for both tasks
* A simple but highly effective **instance-level normalization** to mitigate distribution shifts
* **MEGAugment:** A MEG-oriented [SpecAugment](https://arxiv.org/abs/1904.08779) variant for speech detection
* A **dynamic grouping** loader and **inverse √ class weighting** for phoneme decoding
* Competitive leaderboard results: **88.9% (Speech)**, **65.8% (Phonemes)** (both surpassing baselines)

Checkpoints and training scripts in this repo allow full reproduction of our results.

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

The final scores were obtained using only the 1st seed model.

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

Here, we performed ensembling using the best 5 seeds.

## Trained Checkpoints

The checkpoints used to generate the scores are available here:

https://aholab.ehu.eus/~xzuazo/libribrain/

## Citation

If you use this code or build on MEGConformer, please cite:

```
@misc{dezuazo2025megconformerconformerbasedmegdecoder,
      title={MEGConformer: Conformer-Based MEG Decoder for Robust Speech and Phoneme Classification}, 
      author={Xabier de Zuazo and Ibon Saratxaga and Eva Navas},
      year={2025},
      eprint={2512.01443},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2512.01443}, 
}
```
