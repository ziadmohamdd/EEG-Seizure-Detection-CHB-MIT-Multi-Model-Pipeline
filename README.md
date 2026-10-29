# EEG Seizure Detection on CHB-MIT: Multi-Architecture Benchmark

Code and results for the paper *Accuracy Versus Complexity: A Window-Level Benchmark of Six Deep Architectures for Seizure Detection on CHB-MIT EEG.*

Six deep architectures are trained on band-power features from the
[CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/) under
one protocol, on two feature sets and three optimizers (36 trials). A paired
CNN2D experiment tests whether ECG mean power helps as a network input.

**Scope:** the train/validation/test split is random at window level, so results
describe window-level separability within the same recordings. They are not a
measure of performance on unseen patients.

## Files (`Phases/`)

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

Results are in `Results/` (`results.csv`, `results_ecg_cnn2d.csv`, `leaderboard.xlsx`).

## Data

- **Dataset A:** all retained windows, 90 EEG features (5 bands x 18 bipolar channels).
- **Dataset B:** windows from the 39 records that have an ECG channel. An `ECG_PROXY_mean_power`
  column is stored, but **in the 36-trial benchmark the networks use only the 90 EEG features for
  both datasets.** ECG enters a network only in `03b_CNN2D_ECG_Experiment.ipynb`.
- Windows: 4 s, 2 s step. Absolute Welch band power, 0.5 to 70 Hz. A window is a seizure window if at least 50% overlaps an annotated seizure. Windows with any channel above 500 uV peak-to-peak are dropped.
- Raw `.edf` files and the generated CSVs are not included (size). Regenerate them with notebook 01.
- Dataset A used in the benchmark holds 691,178 windows (691,185 rows minus 7 rows with NaN features). The final preprocessing run reported 1,057,575 windows, so the benchmark file may come from an earlier preprocessing run.
- Dataset B covers 3 patients; only chb04 (74) and chb12 (30) have seizure windows.

## Protocol

70/15/15 random stratified split by window (seed 42); train-only z-scoring; class-weighted
binary cross-entropy with log-odds output-bias initialization (clipped to [-3, 0] for models with
Conformer blocks); Adam, Nadam and AdamW; learning rate 1e-3 (2e-4 for models with Conformer blocks);
early stopping on validation AUC; RBF-SVM on the learned embedding (random 20,000-window training subsample).

`best_val_auc` in `results.csv` is the network's maximum validation AUC over epochs (optimistic).
`sensitivity`, `specificity` and the F1 columns come from the SVM on the test split. They describe different classifiers.

## Results

| Architecture | Best validation AUC (dataset / optimizer) | Mean AUC, A | Mean AUC, B | Training time per trial (min) |
|---|---|---|---|---|
| CNN2D | 0.9714 (B / Adam) | 0.849 | 0.889 | 2.0 to 7.9 |
| Conformer | 0.8449 (A / Adam) | 0.714 | 0.803 | 5.3 to 30.1 |
| BiLSTM | 0.9693 (B / Adam) | 0.868 | 0.930 | 14.8 to 48.5 |
| CNN+LSTM | 0.9885 (B / AdamW) | 0.773 | 0.938 | 12.2 to 44.6 |
| LSTM+Conformer | 0.9892 (B / Adam) | 0.899 | 0.977 | 26.7 to 125.9 |
| Complex Hybrid | 0.9114 (B / Adam) | 0.826 | 0.904 | 22.1 to 149.2 |

- Best validation AUC 0.989 (LSTM+Conformer, Dataset B, Adam). Optimizer choice changes a single architecture's result by up to 0.34, more than most gaps between architectures.
- **ECG experiment (CNN2D, Dataset B, 3 seeds, paired):** ECG mean power changed test AUC by +0.006 (95% CI -0.052 to +0.065). No detectable effect; the experiment is underpowered (16 test seizure windows per seed).
- Mean validation AUC is higher on Dataset B than on A for every architecture, but B is a much smaller, different population (104 seizure windows from two patients), and the benchmark networks did not see the ECG feature. The difference is not evidence for ECG.
- Training time ranges from 2 to 149 minutes per trial. The most expensive model (Complex Hybrid) has a lower best AUC than CNN2D, BiLSTM, CNN+LSTM and LSTM+Conformer.
- Models with Conformer blocks were the least stable (Conformer / A / AdamW ended at AUC 0.5003).

## Preprocessing problems found in CHB-MIT (all fixed in notebook 01)

- Most records contain two channels both named `T8-P8`; MNE renames them `T8-P8-0` / `T8-P8-1`, so name matching silently dropped the channel.
- The ECG channel is labeled `ECG` (36 records), `EKG1-CHIN` (2) or `EKG1-EKG2` (1). `VNS` and `LOC-ROC` are not cardiac.
- Per-patient channel sets differed, which changed the number of feature columns and broke the combined CSV. Features are now written on a fixed 18 x 5 layout; missing channels are zero-filled (chb12_27 to chb12_29 contain none of the 18 derivations).
- Seven rows with missing (NaN) feature values in the Dataset A file are dropped at load time.

## Limitations

Random window-level split with 50% overlapping windows; mixed metrics (network validation AUC vs. SVM test metrics);
very few seizure windows in Dataset B; SVM trained on a subsample; one seed per benchmark trial; recurrent and attention layers
run across the five frequency bands, not across time; average reference applied to bipolar derivations; unequal learning-rate settings across architectures.
