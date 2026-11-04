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

Protocol1 holds out one participant for testing and the next sorted participant
for validation. Both are excluded from every training context. Test uses only
step4. Validation uses step4 for experiments1/2 and steps1–3 for experiment3.
Use `--test-subject 1Alice` for one fold or omit it for all eight.

Protocol2 holds out assembly3 for testing. Whole assemblies from prefixes1/2 are
split into training and validation, with a target validation fraction of 20%.
All objects/sub-actions of an assembly stay together. The same step4 validation
set is used in experiments1/2; experiment2 adds eligible isolated examples to
training. Experiment3 derives validation only from isolated source trials and
never uses step4 for checkpoint selection. All participants may occur on both
sides in this protocol. Splits precede all generated windows/frames.

The audited eight-participant annotation counts (before window generation) are:

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
`--epochs`, `--batch-size`, `--device`, `--workers`, `--run-name`.
Signals additionally support `--points`, IMU `--sensors`, sEMG `--grids`.
Video supports `--model tsm|mvtn`, `--views`, `--frames`, `--image-size`.
Configurations can be passed as full JSON paths or experiment/protocol short names.
The default worker count is 0 for Windows portability.

## Training and outputs

The macro baseline uses AdamW/cross-entropy with optional CUDA AMP, gradient
clipping, ReduceLROnPlateau on validation loss, and early stopping. Best validation
macro F1 selects the checkpoint, with validation loss as tiebreaker. Test is
computed once after that checkpoint is restored. Signal normalization is fitted
on the training partition only. A separate model/scaler is trained for each fold.

Runs have descriptive names containing modality, model, batch, maximum epochs,
experiment, protocol, sensors/grids/views, input size, learning rate and seed.
Repeated default names get numeric suffixes; an explicit existing name is rejected.
Reports include config, annotations, parent splits/exclusions, issues, recording
manifest, imported-source versions, checkpoints, epoch history, predictions,
per-class metrics, confusion matrices and report.html with PNG/SVG plots.
`training.save_windows=true` additionally saves prepared signal arrays; default
false avoids large duplicate array files. Review run/state.json after preparation.

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
in the corresponding preparation run's issues.json and are not auto-repaired.
Bingqian's step4 folder was renamed to rgbd/session4 by the dataset owner.
RGB and depth preparation for the above view both passed after the rename
(72 training, 72 validation and 72 test segments for protocol2). The default
session mapping works directly; no recording override is needed.
No participant or bad interval is silently removed; select a reviewed subset or
resolve source availability before a full recording-based run.

The restricted execution environment denied Windows multi-process worker creation
during a verification run. The default workers 0 training completed; workers 2 was
not validated outside that restriction. Use workers 0 first.
