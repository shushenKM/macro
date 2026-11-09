# Micro-gesture recognition during complete assembly activities

This independent project recognizes seven micro gestures in step4 macro-activity
recordings. It reuses the model architectures and read-only recording/preprocessing
functions from sibling `imu_semg` and `rgb_d` repositories. It does not classify the
whole assembly activity or detect gesture boundaries: input intervals come from
annotations. Neither source project nor the dataset is modified.

## Portable layout

```text
Manipulative_dataset/
  dataset/       # strictly read-only
  imu_semg/      # read-only model/reader dependency
  rgb_d/         # read-only model/reader dependency
  macro/
    train_imu.py, train_semg.py, train_rgb.py, train_depth.py
    check_environment.py
    macro_tools/  # annotations, splits, adapters, training, metrics, plots, paths
    configs/     # modality settings, dataset, experiments, protocols
    requirements.txt
    tests/
    runs/        # every generated artifact and plotting cache
```

All paths resolve from this repository; the path above Manipulative_dataset can
change. Keep all three Git repositories beside dataset on each computer. Source
code imports disable bytecode writes, including in spawned workers. One Python
process activates either imu_semg or rgb_d to avoid their shared `scripts` package
name colliding. Each entry point already starts independently. Run provenance
includes source Git revisions and SHA256 hashes of the actual imported files.

## One environment for all modalities and projects

Python 3.11 is the tested version. Activate your existing CUDA-enabled
`imu_semg_env`, change to this macro directory, then run:

```powershell
python -m pip install -r requirements.txt
python check_environment.py
```

These requirements combine NumPy, SciPy, PyTorch, h5py, OpenCV headless and
Matplotlib for the four implemented training entry points in all three projects.
They do not promise compatibility with every unused script in the original
third-party research repositories. Avoid installing multiple OpenCV variants in
one environment; the headless package includes video/image reading but no GUI.
The existing CUDA build is retained if it satisfies the torch requirement.

For a new NVIDIA environment matching the GPU build already verified on this
computer, install PyTorch before the remaining requirements:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python check_environment.py
```

The CUDA wheel command follows the official PyTorch archive:
https://pytorch.org/get-started/previous-versions/#v270
Use `--device cpu` when CUDA is not available. No environment was replaced or
packages installed automatically during project creation.

## Confirmed annotation interpretation

Step4 contains three complete assembly executions per participant. Each execution
includes three pick-and-screw occurrences for object1, three for object2, three
pick-and-insert occurrences for object3, two for object4 and one for object5.
`1_1`, `1_2`, `1_3` identify occurrences within assembly1; `2_*` and `3_*` identify
assemblies2 and3. Grouping is `(participant, step4, assembly prefix)` across ALL
objects. Raw repetition text and source row identifiers are preserved.

Target classes are approach, grasp, insert, lift, move, release and screw. Isolated
place/deinsert rows are explicitly excluded from mixed/source-only experiments.
Each complete step4 micro-gesture interval supplies one example (including screw).
Steps1–3 retain their existing screw-excerpt preparation. Source times are never
edited and no missing signals are silently padded or substituted.

All current step4 video annotations use the above camera. Front/left/right are
not annotated for step4, so `above` is the default. Additional views are accepted
only if their actual interval annotations exist. Depth samples use saved RGB frame
indices through the existing timestamps.csv mapping, not an assumed acquisition
clock. Depth input retains uint16 values before fixed-range normalization.

## Three experiments, two protocols

| Experiment | Training context | Test context |
|---|---|---|
| step4_to_step4 | Step4 training assemblies | Held-out step4 assemblies/participants |
| steps1234_to_step4 | Same step4 training assemblies plus eligible steps1–3 | Same step4 test set |
| steps123_to_step4 | Steps1–3 only | Same step4 test set |

Protocol1 uses participant-disjoint training, validation and test sets, shared
by IMU, sEMG, RGB and depth. Test uses only step4. Validation uses step4 for
experiments1/2 and steps1–3 for experiment3. Test/validation participants are
excluded from **every** training context, including steps1–3 in mixed training.

Edit `configs/protocols/protocol1.json` to choose participant counts:

```json
{
  "name": "protocol1",
  "split_before_windowing": true,
  "type": "subject_holdout",
  "subject_counts": {"train": null, "validation": 1, "test": 1},
  "test_subject_ids": null,
  "validation_subject_ids": null
}
```

- `train: null` uses all participants left after validation and test selection.
  With eight participants the default is **6 train / 1 validation / 1 test**.
  Set an integer to use fewer training participants; unused participants are
  excluded from all partitions and listed in the saved metadata.
- `test: 1` and `test_subject_ids: null` run **all eight leave-one-subject-out
  folds**, one independent model per test participant. If `--subjects` filters
  the population, there is one fold per selected participant instead.
- Automatic validation takes the next available participant(s), in cyclic sorted
  order after the first test participant. Training takes the remaining people
  in that order. Counts must fit the selected population, with at least one
  training and one test participant. `validation: 0` uses the last epoch and
  disables validation-based checkpoint selection and scheduling.
- To fix two test participants, set `test: 2` and
  `test_subject_ids: ["1Alice", "2Bingqian"]` (example IDs; choose your pair).
  This runs **one model**, with both participants held out together; it does not
  enumerate all participant pairs. With one validation participant, five remain
  for training. Optionally set `validation_subject_ids` and its matching count.
  Fixed validation IDs require fixed test IDs so that a validation person cannot
  conflict with one of the automatic LOSO test folds.

The same options are available on the command line without editing JSON:

```powershell
# All eight LOSO folds: 6 training, 1 validation, 1 test participant each.
python train_imu.py --experiment step4_to_step4 --protocol protocol1 --train-subject-count 6 --validation-subject-count 1 --test-subject-count 1 --sensors E45A E477 --epochs 50 --device cuda

# One fixed two-participant test set (example pair), with automatic 5/1/2 counts.
python train_imu.py --experiment step4_to_step4 --protocol protocol1 --test-subjects 1Alice 2Bingqian --sensors E45A E477 --epochs 50 --device cuda

# Inspect assignments using annotations only; no signal processing or training.
python train_imu.py --experiment step4_to_step4 --protocol protocol1 --test-subjects 1Alice 2Bingqian --validation-subjects 3Chawki --audit-only
```

Plural CLI ID lists infer their counts unless an explicit `--*-subject-count`
override is also supplied; if supplied, it must match the list length. The legacy
`--test-subject 1Alice` and `--validation-subject 3Chawki` flags still work for a
fixed single-subject split. Counts and participant IDs are saved in the compact
`preparation` section of `config.json` and printed before training.

After all folds complete, the terminal and `fold_summary.txt` display every
fold's test accuracy/F1 and the **unweighted fold mean +/- population standard
deviation**. For eight LOSO folds, this is an equal-participant average, not an
average of training accuracies. Numeric summaries are in `aggregate_metrics.json`;
the root `report.html` shows the summary and a fold comparison plot. A fixed
two-participant test group is one fold, so it has no cross-fold SD estimate.

The original `protocol2` holds out assembly3 for testing. Whole assemblies from prefixes1/2 are
split into training and validation, with a target validation fraction of 20%.
All objects/sub-actions of an assembly stay together. The same step4 validation
set is used in experiments1/2; experiment2 adds eligible isolated examples to
training. Experiment3 derives validation only from isolated source trials and
never uses step4 for checkpoint selection. All participants may occur on both
sides in this protocol. Splits precede all generated windows/frames.

The audited eight-participant annotation counts for that original protocol
(before window generation, using all eligible source data) are:

| Experiment | Protocol2 train | Validation | Test |
|---|---:|---:|---:|
| step4_to_step4 | 936 | 216 | 576 |
| steps1234_to_step4 | 2664 | 216 | 576 |
| steps123_to_step4 | 1376 | 352 | 576 |

For one participant, protocol2 currently assigns one complete assembly to each
of train, validation and test (72 annotated micro gestures each). Accepted time
interval overlaps remain warnings. Duplicate source IDs and assemblies crossing
partitions remain errors. Macro does not perform exhaustive content hashing of
copied signals; it preserves the user's accepted source-overlap policy.

### Protocol 2 with configurable percentages

Select `--protocol protocol2_percentages` to use
`configs/protocols/protocol2_percentages.json`:

```json
"split_percentages": {
  "train": 70,
  "validation": 15,
  "test": 15
}
```

These values must sum to 100. Training and testing must be positive; validation
may be zero. `split_seed` controls the reproducible shuffle. Any assembly ID can
be assigned to any partition; repetitions 1/2/3 no longer determine membership.
Complete `(participant, step4, assembly)` groups, including every object's
sub-actions, stay together. Unequal numbers of object repetitions do not split
an assembly. Groups are pooled across the selected participants, rather than
enforcing the percentages separately within each participant. For unseen-person
evaluation, use protocol1.

Percentages target **assembly counts**. Integer allocation uses largest remainder
with at least one assembly in each nonzero partition. With 24 assemblies, the
70/15/15 request yields 17/4/3 assemblies (70.83/16.67/12.50%). With only one
participant's three assemblies it yields 1/1/1. Different assembly sizes can also
make annotation percentages differ; the terminal and saved split metadata show
both actual assembly counts and annotation percentages. Class counts need not
be equal, and missing partition classes are checked by the existing validator.

For `step4_to_step4`, all three sets follow this step4 split. For
`steps1234_to_step4`, the same step4 sets are used, then eligible steps1–3 are added
only to training; `--steps123-percent` still controls that source contribution.
Thus the percentages do not describe the final combined sample totals.

For `steps123_to_step4`, only the same step4 **test** partition is used. Its planned
step4 training/validation groups are explicitly excluded. Training and validation
come from steps1–3, with the separate `source_validation.fraction` setting (default
0.2) controlling source validation before `--steps123-percent` is applied. Step4
is never used for checkpoint selection in this source-only experiment.

```powershell
python train_semg.py --experiment step4_to_step4 --protocol protocol2_percentages --audit-only
python train_semg.py --experiment steps1234_to_step4 --protocol protocol2_percentages --steps123-percent 50 --grids grid1 grid2 --epochs 50 --device cuda
```

The same protocol works for IMU, sEMG, RGB and depth. Matching annotations, subject
selection and split settings yield matching assembly assignments. Run names
include the requested percentages and split seed. The original `protocol2.json`
remains available for reproducing the earlier fixed-repetition evaluation.

## Begin with annotation audit, then prepare

```powershell
python train_imu.py --experiment step4_to_step4 --protocol protocol2 --audit-only
python train_semg.py --experiment step4_to_step4 --protocol protocol2 --subjects 1Alice --prepare-only
python train_rgb.py --experiment step4_to_step4 --protocol protocol2 --subjects 1Alice --views above --prepare-only
python train_depth.py --experiment step4_to_step4 --protocol protocol2 --subjects 1Alice --prepare-only
```

`--audit-only` reads only annotations and reports whole-trial assignments.
`--prepare-only` additionally checks source recordings and prepares model inputs.
Neither performs optimizer updates. Full training is enabled by omitting both:

```powershell
python train_semg.py --experiment step4_to_step4 --protocol protocol2 --grids grid1 grid2 --epochs 50 --device cuda
python train_imu.py --experiment steps123_to_step4 --protocol protocol1 --sensors E45A E477 --epochs 50 --device cuda
python train_rgb.py --experiment steps1234_to_step4 --protocol protocol2 --model mvtn --views above --epochs 50 --device cuda
python train_depth.py --experiment step4_to_step4 --protocol protocol1 --model tsm --views above --epochs 50 --device cuda
```

Common overrides: `--subjects`, `--test-subject`, `--validation-subject`,
`--epochs`, `--batch-size`, `--device`, `--workers`, `--run-name`, `--amp` / `--no-amp`.
Signals additionally support `--points`, IMU `--sensors`, sEMG `--grids`.
Video supports `--model tsm|mvtn`, `--views`, `--frames`, `--image-size`.
Configurations can be passed as full JSON paths or experiment/protocol short names.
The default worker count is 0 for Windows portability.

## Limit steps 1–3 training data

Both source-data experiments accept `--steps123-percent`, shared by all four
modality entry points and both protocols. For example:

```powershell
python train_semg.py --experiment steps1234_to_step4 --protocol protocol2 --steps123-percent 50 --grids grid1 grid2 --epochs 50 --device cuda
python train_semg.py --experiment steps123_to_step4 --protocol protocol2 --steps123-percent 25 --grids grid1 grid2 --epochs 50 --device cuda
```

The percentage applies to eligible **steps1–3 training trials**, after participant
holdouts and validation assignment. It is not the percentage of the combined
training dataset. Step4 training records and all validation/test records stay
fixed. In `steps123_to_step4`, validation still uses the original source validation
partition; only the remaining source training pool is reduced.

The default is 100 (all eligible source training data). You can instead set
`"steps123_train_percent": 50` in either corresponding JSON under
`configs/experiments/`; the command-line value overrides it. Zero is supported
only for `steps1234_to_step4`, leaving its step4 training data. Pure source training
requires a positive percentage. `step4_to_step4` has no source data to reduce.

Sampling uses the training seed, stratifies by participant/session/gesture
composition, and keeps entire isolated trials together before window generation.
Each nonempty stratum retains at least one trial for any positive percentage.
Rounding, small strata, unequal trial sizes and subsequent screw window expansion
mean the final sample percentage can differ from the requested trial percentage.
The same seed and annotations yield the same selected trials across modalities.

The terminal prints eligible/retained annotation and trial counts. Run names include
`src50pct` (or the chosen percentage); `config.json` records the request, seed,
and actual counts. Full selected-trial and excluded-record lists stay in memory.
Use `--audit-only` to inspect the selection summary without
loading signals or starting training.

## Training and outputs

The macro baseline uses AdamW/cross-entropy with optional CUDA AMP, gradient
clipping, ReduceLROnPlateau on validation loss, and early stopping. Best validation
macro F1 selects the checkpoint, with validation loss as tiebreaker. Test is
computed once after that checkpoint is restored. Signal normalization is fitted
on the training partition only. A separate model/scaler is trained for each fold.

sEMG defaults to FP32, including on CUDA. A step4 validation segment produced
finite normalized values around 289,240 that overflowed the first convolution
under FP16 AMP. FP32 handles that range without clipping or excluding the segment.
Use `--no-amp` to select FP32 explicitly; this still uses the GPU with `--device cuda`.
Other modalities retain their existing AMP defaults. The terminal and the
`training_details.environment` section of `result.json` record the effective precision. Extreme signal values still merit source
review; changing precision does not repair the recording.

Runs have descriptive names containing modality, model, batch, maximum epochs,
experiment, protocol, sensors/grids/views, input size, learning rate and seed.
Repeated default names get numeric suffixes; an explicit existing name is rejected.
Outputs focus on parameters and results:

- `config.json`: effective parameters, status, labels, compact split/subject counts,
  warning counts, and imported-source versions.
- `results.json`, `aggregate_metrics.json`, `fold_summary.txt`, and `report.html`:
  completed-fold results, cross-validation summaries, and links to the plots.
- Each trained fold: `best.pt` / `last.pt`, `history.csv`, `result.json`,
  `predictions.csv`, `confusion_matrix.csv`, `report.html`, and `plots/`.
  `result.json` includes test metrics and model/device/training details.

Training no longer exports annotations, parent splits, recording manifests,
per-sample lists, separate labels/normalization/model/environment JSON files,
or plot-status files. Labels and normalization remain inside checkpoints.
Preparation/audit-only runs save their summary in `config.json`. Failures alone
create `error.json`; a plotting failure creates `plot_error.txt`. Existing runs
are not cleaned or changed automatically, and old reports can still be regenerated.

`training.save_windows=true` additionally saves prepared signal arrays; default
false avoids large duplicate array files. Review `config.json` after preparation.

```powershell
python -B -m macro_tools.plots --run "runs/semg/your_completed_run"
python -B -m unittest discover -s tests -v
```

Synthetic checks are labelled as such. A one-epoch chart shows actual values as
bars because no learning curve can be inferred from a single epoch.

## Verification and current source-data limitations

Tests cover compound annotations, whole-assembly isolation across objects,
identical test populations, training-only normalization, source adapters,
checkpoint selection/test isolation, and plot generation. Small real-data GPU
checks completed for Alice's sEMG and RGB; Alice's RGB/depth preparation passed.
These one-epoch runs are software checks, not research performance estimates.

All annotation-only scenarios/protocols passed with eight participants. Full
recording preparation is a separate check: Alice's selected IMU intervals include
71 intervals outside the available recording, two intervals too short for the
configured filter, and one interval with insufficient samples. These are saved
in the corresponding historical preparation run's issues.json (new failed runs
use error.json) and are not auto-repaired.
Bingqian's step4 folder was renamed to rgbd/session4 by the dataset owner.
RGB and depth preparation for the above view both passed after the rename
(72 training, 72 validation and 72 test segments for protocol2). The default
session mapping works directly; no recording override is needed.
No participant or bad interval is silently removed; select a reviewed subset or
resolve source availability before a full recording-based run.

The restricted execution environment denied Windows multi-process worker creation
during a verification run. The default workers 0 training completed; workers 2 was
not validated outside that restriction. Use workers 0 first.


## Standard TCN for IMU and sEMG

Add `--model tcn` to the existing signal training commands. Macro imports the
shared TCN from the sibling `imu_semg` repository through its read-only adapter.
Keep both repositories updated and beside each other. Dataset, preprocessing,
experiment/protocol splits, optimizer, scheduler, batch size, AMP setting,
early stopping and reporting keep their existing macro settings. Only the
architecture and its model-specific parameters change. Existing transformers
remain the defaults; `--model transformer` selects them explicitly.

```powershell
python train_imu.py --experiment step4_to_step4 --protocol protocol2 --model tcn --sensors E45A E477 --epochs 50 --device cuda
python train_semg.py --experiment step4_to_step4 --protocol protocol2 --model tcn --grids grid1 grid2 --epochs 50 --device cuda
```

The same model flag works with `steps1234_to_step4`, `steps123_to_step4`,
`protocol1`, `protocol2_percentages`, subject-count/ID options and
`--steps123-percent`. Add `--prepare-only` to prepare without training. One-sensor
and one-grid selection are supported as before. RGB/depth model choices remain
TSM/MVTN.

For customized TCN architecture parameters use `--config configs/imu_tcn.json`
or `--config configs/semg_tcn.json`. Each file initially copies its corresponding
macro base config and replaces only the model fields; it is independent of later
base-config edits. `--model tcn` without a custom config instead uses the current
base settings. Switching architecture with `--model` resets `model_parameters`
to the selected architecture's defaults; selecting the same architecture keeps
custom parameters.

TCN defaults: six residual blocks, 64 channels per block, two causal convolutions
per block, kernel size 3, dilations 1/2/4/8/16/32, dropout 0.2, then temporal mean
pooling and a classification head. IMU inputs retain nine derived features per
sensor; sEMG retains the existing envelope preprocessing and flattens the selected
grids into 64/128 electrode time series. This is standard TCN without spatial
8x8 convolutions. It is not the separate spatial CNN-TCN architecture.
Architecture and parameters are recorded in config.json and checkpoints, and
TCN is included in run folder names. Outputs remain inside macro/runs/.
