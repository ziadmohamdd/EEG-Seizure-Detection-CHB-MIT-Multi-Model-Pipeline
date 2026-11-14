# EEG Seizure Detection on CHB-MIT: Multi-Architecture Benchmark

Code and results for the paper *Epileptic Seizure Detection in Imbalanced Scalp EEG: A Comparative Analysis of Advanced Deep Learning Architectures* (A. Solia, E. Dwedar, Z. Mohamed, A. Anter), accepted at JAC-ECC 2026.

Six deep architectures are trained on band-power features from the
[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) under
one protocol, on two datasets with three optimizers (36 trials). Each trial is scored three ways:
the network's validation AUC, the network's exact test AUC and AUPRC restored from its best checkpoint,
and an RBF-SVM on the learned embedding. A paired CNN2D experiment tests whether ECG mean power helps
as a network input.

**Scope:** the train/validation/test split is random over 50%-overlapping windows, so overlapping windows
of the same seizure appear in training and test sets. Results describe window-level separability within
the same recordings. They are not a measure of performance on unseen patients.

## Files

### `Phases/`

| File | Purpose |
|---|---|
| `01_Phase1_Preprocessing.ipynb` | `.edf` records to windowed band-power features (`dataset_A.csv`, `dataset_B.csv`) |
| `02_Utils_ModelUtils.ipynb` | Shared helpers (reference only; each model notebook carries its own copies) |
| `03_Model1_CNN2D.ipynb` | CNN2D |
| `03b_CNN2D_ECG_Experiment.ipynb` | Paired test of ECG mean power as a second network input |
| `04_Model2_Conformer.ipynb` | Conformer |
| `05_Model3_LSTM.ipynb` | Bidirectional LSTM |
| `06_Model4_CNN_LSTM.ipynb` | CNN + LSTM |
| `07_Model5_LSTM_Conformer.ipynb` | LSTM + Conformer |
| `08_Model6_ComplexHybrid.ipynb` | Three-branch CNN / LSTM / Conformer hybrid |
| `09_Phase3_ResultsCompilation.ipynb` | Builds the leaderboard from `results.csv` |
| `10_Checkpoint_Test_Evaluation.ipynb` | Restores the best checkpoint of all 36 trials and computes exact test AUC and AUPRC on the same seed-42 test split (no retraining) |

### `Results/`

| File | Content |
|---|---|
| `results.csv` | One row per trial: validation AUC (network) and SVM test metrics, training time |
| `network_test_results_36_trials.csv` | One row per trial: exact test AUC and test AUPRC of the restored network (from notebook 10) |
| `results_ecg_cnn2d.csv` | Paired ECG experiment, three seeds |
| `leaderboard.xlsx` | Leaderboard built by notebook 09 |

## How to reproduce

1. Download CHB-MIT from PhysioNet and run `01` to build `dataset_A.csv` and `dataset_B.csv`.
2. Run `03` to `08` to train the 36 trials. Each notebook saves its best checkpoints to `OUTPUT_DIR/checkpoints/` and appends to `results.csv`.
3. Run `10` to compute the network test AUC and AUPRC from the saved checkpoints.
4. Run `03b` for the paired ECG experiment and `09` for the leaderboard.

All notebooks were run on Google Colab with an NVIDIA Tesla T4 GPU (16 GB). Raw `.edf` files, the generated CSVs and the checkpoints are not included (size).

## Data

- **Dataset A:** retained windows with 90 EEG features (5 bands x 18 bipolar channels). The file used in the benchmark has 691,185 rows; 7 rows with NaN features are dropped at load time, leaving 691,178 windows. The final preprocessing run wrote 1,057,575 windows, so the benchmark file comes from a different run of the same pipeline. It has no all-zero feature column, so it includes the T8-P8 fix.
- **Dataset B:** windows from the 39 records that have an ECG channel (164,386 windows). An `ECG_PROXY_mean_power` column is stored, but **in the 36-trial benchmark the networks use only the 90 EEG features for both datasets.** ECG enters a network only in `03b_CNN2D_ECG_Experiment.ipynb`.
- Dataset B covers 3 patients; only chb04 (74) and chb12 (30) have seizure windows. **All 30 chb12 seizure windows have all-zero EEG features** (see below), so the real EEG seizure information in Dataset B comes from chb04 alone.
- Windows: 4 s, 2 s step (50% overlap). Absolute Welch band power, 0.5 to 70 Hz. A window is a seizure window if at least 50% of it overlaps an annotated seizure.
- Windows with any channel above 500 uV peak-to-peak are dropped. In the final preprocessing run this removed 711,004 of 1,768,579 windows (40.2%), including 4,043 windows that overlapped a seizure.

## Protocol

70/15/15 random stratified split by window (seed 42); train-only z-scoring; class-weighted
binary cross-entropy with log-odds output-bias initialization (clipped to [-3, 0] for models with
Conformer blocks); Adam, Nadam and AdamW; learning rate 1e-3 (2e-4 for models with Conformer blocks);
early stopping on validation AUC; RBF-SVM on the learned embedding (random 20,000-window training subsample).

Three kinds of numbers are reported, and they come from different places:

- `best_val_auc` (`results.csv`): the network's maximum validation AUC over epochs, as computed by Keras on 200 fixed thresholds. Optimistic, and binned.
- `test_auc`, `test_auprc` (`network_test_results_36_trials.csv`): the restored network on the test split, exact (rank-based) AUC and average precision.
- `sensitivity`, `specificity`, F1 columns (`results.csv`): the SVM on the test split. A different classifier built on the same embedding.

## Results

| Architecture | Best validation AUC (dataset / optimizer) | Mean val. AUC, A | Mean val. AUC, B | Mean test AUC, A | Mean test AUC, B | Training time per trial (min) |
|---|---|---|---|---|---|---|
| CNN2D | 0.9714 (B / Adam) | 0.849 | 0.889 | 0.845 | 0.780 | 2.0 to 7.9 |
| Conformer | 0.8449 (A / Adam) | 0.714 | 0.803 | 0.862 | 0.769 | 5.3 to 30.1 |
| BiLSTM | 0.9693 (B / Adam) | 0.868 | 0.930 | 0.913 | 0.893 | 14.8 to 48.5 |
| CNN+LSTM | 0.9885 (B / AdamW) | 0.773 | 0.938 | 0.762 | 0.949 | 12.2 to 44.6 |
| LSTM+Conformer | 0.9892 (B / Adam) | 0.899 | 0.977 | 0.908 | 0.969 | 26.7 to 125.9 |
| Complex Hybrid | 0.9114 (B / Adam) | 0.826 | 0.904 | 0.824 | 0.951 | 22.1 to 149.2 |

- **Best model:** LSTM+Conformer (Dataset B, Adam) reached a validation AUC of 0.9892 and a test AUC of 0.9889. It ranked first in the Friedman test on both validation AUC (p = 0.003) and test AUC (p = 0.027).
- **Validation vs. test:** test AUC tracks validation AUC (Spearman rho = 0.80, p < 0.001). The largest gap is Conformer / A / AdamW: validation AUC 0.5003 (binned Keras AUC) but exact test AUC 0.823.
- **Optimizer:** changes a single architecture's validation AUC by up to 0.34, more than most gaps between architectures. Adam ranks first; the difference is significant on validation AUC (p = 0.006) but not on test AUC (p = 0.076).
- **Dataset A vs. B:** validation AUC is higher on B in 16 of 18 pairs (p < 0.001), but test AUC only in 12 of 18 (p = 0.21). B is a much smaller population with 30 all-zero seizure windows, and the benchmark networks did not see the ECG feature, so the difference is not evidence for ECG.
- **Clinical operating point:** test AUPRC is 0.005 to 0.346 (chance: 0.0020 on A, 0.00065 on B). At the SVM operating point, precision is 0.5% to 10.5% and false-positive windows per hour range from 18 to 452, so none of these window-level detectors is close to clinical use.
- **ECG experiment (CNN2D, Dataset B, 3 seeds, paired):** ECG mean power changed test AUC by +0.006 (95% CI -0.052 to +0.065, p = 0.69). No detectable effect; the experiment is underpowered (16 test seizure windows per seed).
- **Cost:** training time ranges from 2 to 149 minutes per trial. The most expensive model (Complex Hybrid) has a lower best validation AUC than CNN2D, BiLSTM, CNN+LSTM and LSTM+Conformer.
- **Stability:** models with Conformer blocks were the least stable during training.

## Preprocessing problems found in CHB-MIT (all fixed in notebook 01)

- **Duplicate channel name.** 655 of 664 records contain two channels both named `T8-P8`; MNE renames them `T8-P8-0` / `T8-P8-1`, so a lookup by name found no `T8-P8` and the channel was silently zero-filled. The first copy is now mapped back to `T8-P8`.
- **ECG labels.** The ECG channel is labeled `ECG` (36 records), `EKG1-CHIN` (2) or `EKG1-EKG2` (1). Matching only `EKG1-CHIN` kept 1,257 windows (30 seizure) instead of 164,386. `VNS` and `LOC-ROC` are not cardiac.
- **Varying channel sets.** Per-patient channel sets differed, which changed the number of feature columns and broke the combined CSV. Features are now written on a fixed 18 x 5 layout with missing channels zero-filled. chb12_27 to chb12_29 contain none of the 18 derivations, so their EEG features are all zero. Dataset B's only chb12 records are chb12_28 and chb12_29, so all 30 chb12 seizure windows (29 + 1) and 1,257 rows of Dataset B are all zero.
- **Missing values.** Seven rows with NaN feature values in the Dataset A file are dropped at load time; left in, they make the loss NaN from the first batch.

## Limitations

- Random window-level split over 50%-overlapping windows: overlapping windows of the same seizure fall into training and test sets (temporal leakage). Results do not measure performance on unseen patients.
- Validation AUC is binned and maximized over epochs; SVM metrics come from a different classifier than the network AUC and AUPRC.
- Dataset B has 104 seizure windows from two patients, 30 of them all-zero, and 16 per test split.
- The average reference was applied to bipolar derivations and included the ECG and other non-EEG channels; the 500 uV threshold removed 4,043 seizure-overlapping windows.
- One seed per benchmark trial; the SVM is trained on a 20,000-window subsample.
- Recurrent and attention layers run across the five frequency bands of one window, not across time; Conformer blocks have no positional encoding; no model encodes electrode geometry.
- Models with Conformer blocks used a different learning rate and bias clipping.
- Two trials (CNN+LSTM B/Nadam and LSTM+Conformer B/Nadam) were resumed after a runtime interruption; their logs and checkpoints cover only the final session.

## Citation

```bibtex
@inproceedings{solia2026seizure,
  author    = {Solia, Ahmed and Dwedar, Esraa and Mohamed, Ziad and Anter, Ahmed},
  title     = {Epileptic Seizure Detection in Imbalanced Scalp {EEG}: A Comparative Analysis of Advanced Deep Learning Architectures},
  booktitle = {Proc. International Japan-Africa Conference on Electronics, Communications and Computations (JAC-ECC)},
  year      = {2026},
  note      = {Accepted}
}
```

The CHB-MIT database should be cited as described on its [PhysioNet page](https://physionet.org/content/chbmit/1.0.0/).
