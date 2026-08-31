# Complexity-routed PINN/NDE prototypes

This directory is a clean reimplementation of the Morse-routed experiments in
`phd_thesis/complexity_estimation`. It provides two low-capacity expert choices
(RBF and Fourier) and an MLP high-capacity expert for both PINNs and neural
differential equations.

## Design

The model is

```text
input -> fixed analytic Morse score -> fitted soft gate
      -> RBF/Fourier simple expert ----\
                                       weighted assembled output
      -> MLP complex expert -----------/
```

Important corrections relative to the original scripts:

- the routing threshold is fitted on training points once and saved as model state;
- RBF, Fourier, gate and ensemble operations stay in PyTorch and remain differentiable;
- PINN residuals are evaluated on the assembled prediction over the whole domain;
- expert agreement is enforced at the soft routing interface, at identical points;
- NDE training constrains the assembled vector field and uses a smooth gate;
- the NDE example is non-polynomial, so its simple expert does not contain the exact answer;
- a complex-usage penalty is available for later accuracy/compute trade-off studies.

`HeterogeneousEnsemble` only requires experts mapping an input tensor to an
output tensor. That boundary is intentional: a windowed/flow-map SSM can later
be introduced without changing routing, losses, or experiment reporting. A
stateful SSM cannot be used as an instantaneous ODE right-hand side without an
augmented hidden state, so that migration should be treated separately.

## Run

From this directory:

```bash
python -m unittest discover -s tests -v
python -m examples.train_heat --simple rbf --epochs 1000
python -m examples.train_heat --simple fourier --epochs 1000
python -m examples.train_pendulum --simple rbf --epochs 1000
python -m examples.train_pendulum --simple fourier --epochs 1000
```

The examples automatically use CUDA when it is available. The training scripts
are deliberately minimal; repeated-seed comparisons and measured runtime/FLOP
accounting should be added after this architectural baseline is validated.

## Metrics and visualizations

Each training command writes two files to `artifacts/` by default:

- `<problem>_<expert>.png`: prediction and reference comparisons, absolute
  errors, Morse complexity, routing behavior, phase portrait where applicable,
  and loss curves;
- `<problem>_<expert>_metrics.json`: machine-readable quality and routing
  metrics.

The heat PINN reports relative L2, MSE, MAE, maximum error, full-domain physics
residual MSE, interface disagreement, mean soft complex weight, and hard-route
fraction. The pendulum NDE reports trajectory relative L2/RMSE/MAE, final-state
error, vector-field MSE, kinematic residual, interface disagreement, and routing
usage. Use `--output-dir PATH` to change the artifact directory.

## Baseline and robustness comparisons

The controlled comparison uses three models:

1. `morse`: the proposed Morse-routed RBF/Fourier + selected complex ensemble;
2. `learned`: the same two experts with a conventional learned input gate and
   the same target complex-expert fraction;
3. `single_complex`: the selected complex expert on its own (MLP or PINNMamba).

For each seed, all candidates receive the same sampled data and noise. Training
noise and inference noise are independent axes. Every model is trained with both
clean and Gaussian-corrupted targets, then evaluated with both clean and noisy
observations. This includes the requested clean-training/noisy-inference case.
The output CSV records the complex expert used, both noise levels, clean-domain error, OOD error,
physics/vector-field error, parameter count and routing usage for every
individual run. The PNG summarizes means, seed-to-seed standard deviations and
the degradation caused independently by training and inference noise.

```bash
python -m examples.compare_heat --simple rbf --epochs 500 --seeds 0,1,2 --training-noise 0.05 --inference-noise 0.05
python -m examples.compare_heat --simple fourier --epochs 500 --seeds 0,1,2 --training-noise 0.05 --inference-noise 0.05
python -m examples.compare_pendulum --simple rbf --epochs 500 --seeds 0,1,2 --training-noise 0.05 --inference-noise 0.05
python -m examples.compare_pendulum --simple fourier --epochs 500 --seeds 0,1,2 --training-noise 0.05 --inference-noise 0.05
```

For the heat equation, OOD means extrapolation from training times `[0,1]` to
`[1,1.25]`. For the pendulum, OOD means rollout from a larger unseen initial
condition. PINN inference noise perturbs the coordinates presented to the model,
but predictions are scored against targets at the original clean coordinates.
NDE inference noise perturbs the state observed by the learned vector field at
every RK4 substep, while the reference trajectory remains clean. Noise values
are absolute Gaussian standard deviations in the normalized/model coordinates.
These are robustness experiments, not adversarial certificates.

## PINNMamba reproduction baseline

`complexity_ensemble.pinnmamba` is a pure-PyTorch reimplementation of the
official ICML 2025 PINNMamba architecture. It preserves the selective SSM,
trainable sine/cosine activation, seven-step forward time subsequences, and
overlap-alignment loss. The default model has exactly 285,763 parameters, as
reported in the paper. It has no custom CUDA dependency, which also makes it a
clean modular expert for the next ensemble stage.

The first reproduction target is the paper's 1D reaction equation. To evaluate
an official or locally trained checkpoint and generate metrics plus a dashboard:

```bash
python -m examples.reproduce_pinnmamba_reaction --checkpoint PATH/1d_reaction_PINNMamba_7_0.01.pt
```

To train from scratch with the released-code setup (101 x 101 grid, L-BFGS,
alignment weight 1000):

```bash
python -m examples.reproduce_pinnmamba_reaction --steps 500
```

The released reaction script uses 500 L-BFGS outer steps and boundary weight 1,
while the paper's training-details paragraph says 1000 epochs and boundary
weight 10. The CLI exposes `--steps` and `--boundary-weight` so these two
configurations are not silently conflated. The paper comparison targets are
rMAE 0.0094 and rRMSE 0.0217.

### Morse-routed PINNMamba ensemble

PINNMamba can replace the complex MLP expert without changing the RBF/Fourier
branch, Morse/learned router, quality metrics, noise protocol, or dashboards:

```bash
python -m examples.train_heat --simple rbf --complex pinnmamba --epochs 1000
python -m examples.compare_heat --simple rbf --complex pinnmamba --epochs 500 --seeds 0,1,2
```

The adapter exposes pointwise anchor predictions to the existing evaluators,
but training evaluates the heat-equation residual over every sequence token.
Overlap consistency is imposed on the final routed mixture, so the simple and
PINNMamba experts describe one continuous solution across adjacent windows.
The default overlap weight is 1000, matching the released reaction setup.
For memory-safe second derivatives, each optimization step samples sequence
tokens and computes the exact diagonal PDE derivative at those positions; this
avoids the cross-token Jacobian sum present in the released training script.

## Switzerland/Ukraine rainfall-runoff benchmark

The environmental benchmark predicts next-day specific discharge from a
30-day window of precipitation, temperature, PET, previous discharge, and
seasonal coordinates. CAMELS-CH can additionally include its annual crop,
grass, forest, urban, and ice fractions. Splits are chronological (60/20/20),
normalization is fitted on the training split only, and windows never cross a
missing calendar day.

The controlled comparison is unchanged in spirit:

1. `morse`: RBF/Fourier + selected complex expert with an analytic hydrologic
   Morse gate;
2. `learned`: the identical experts with a learned gate;
3. `single_complex`: the same MLP or PINNMamba complex expert alone.

All candidates also use the same weak first-order reservoir consistency loss,
`dQ/dt = response * max(P - PET, 0) - recession * Q`. This is a deliberately
small physics prior, not a replacement for a calibrated distributed
hydrological model. The PINNMamba adapter reuses the existing selective SSM and
wave activations but accepts hydrological feature sequences instead of `(x,t)`
coordinates. RBF/Fourier and MLP receive the exact same window flattened.

Run the locally available Swiss basin with either complex architecture:

```bash
python -m examples.compare_hydrology --source camels_ch --basin 2011 --simple rbf --complex mlp
python -m examples.compare_hydrology --source camels_ch --basin 2011 --simple fourier --complex pinnmamba
```

To train and report metrics for only the analytic Morse model, select it and
provide multiple seeds for the reported mean and standard deviation:

```bash
python -m examples.compare_hydrology \
  --source camels_ch --basin 2011 --models morse \
  --simple rbf --complex mlp --epochs 100 --seeds 0,1,2,3,4
```

For Ukraine, prepare a licensing-compliant, daily aligned CSV with columns
`date`, `discharge_mm_day`, `precipitation_mm_day`, `temperature_c`, and
`pet_mm_day`, then run:

```bash
python -m examples.compare_hydrology --source ukraine_csv --csv PATH.csv --basin GAUGE_ID
```

The CSV adapter also accepts a programmatic column-role mapping for native
EStreams/GRDC export names. Raw GRDC observations are intentionally not copied
or downloaded by this project. Output CSVs report validation/test NSE, test
KGE and RMSE, reservoir residual, parameter count, routing usage, and both
noise axes; the PNG compares every requested training/inference noise condition.

Annual CAMELS-CH land cover is useful for sensitivity and temporal-shift
experiments, but its interpolated covariates do not by themselves identify a
causal effect of land-cover change. Treat counterfactual attribution as a
separate validation stage.

### Testing the Morse routing hypothesis

The routing diagnostic tests whether a trained hydrology ensemble's fixed
Morse score is actually aligned with where routing is useful:

```bash
python -m examples.diagnose_hydrology_routing \
  --source camels_ch --basin 2011 --simple rbf --complex mlp --epochs 20
```

It reports Pearson and Spearman correlations between the score and simple
expert error, complex-expert error advantage, reservoir-residual advantage,
discharge, and absolute flow change. It also measures the precision with which
the top-scored 20% recovers the 20% of samples with the largest complex-expert
advantage; 0.20 is the random-overlap reference. Per-sample CSVs and a scatter
plot are saved so misleading correlations driven by a few flood events can be
inspected directly. Because the experts are trained within the routed model,
these diagnostics describe the current ensemble and should not be interpreted
as a causal estimate of intrinsic expert difficulty.

### Data-calibrated Morse potential

`learned_morse` retains the Morse construction but learns its scalar potential
before final ensemble training. Forward-chaining pilot models produce
out-of-sample labels

```text
simple loss - complex loss
+ 0.05 * (simple reservoir residual - complex reservoir residual)
```

and a small scalar `tanh` network is trained so the norm of its input gradient
ranks windows in the same order as this complex-expert utility. The potential
uses differentiable last/mean/variability/change/3-day/7-day summaries of each
input feature. It is frozen before its 80th-percentile routing threshold and
the final ensemble are fitted.

Run the four-way analytic-Morse, learned-Morse, learned-gate and unitary
comparison with:

```bash
python -m examples.compare_learned_morse_hydrology \
  --source camels_ch --basin 2011 --simple rbf --complex mlp \
  --folds 3 --pilot-epochs 5 --potential-epochs 100 --epochs 20 \
  --training-noise 0.1 --inference-noise 0.1
```

The learned-Morse comparison trains clean and noisy-input copies of every final
model and evaluates each copy on clean and noisy inputs. Noise is Gaussian in
training-standardized units and affects precipitation, temperature, PET, and
previous discharge only. Targets, static land cover, and calendar coordinates
remain clean. The resulting four train/inference conditions are recorded in
the comparison CSV and printed in the summary.

Calibration uses only the main training period; validation and test remain
untouched. The command saves the potential checkpoint, calibration statistics,
final comparison CSV, and comparison plot. This candidate has an explicit
pilot-training cost, which must be reported alongside accuracy when comparing
it with the analytic router.

## Universal differential equation rainfall-runoff model

The UDE experiment is separate from the windowed sequence predictor. It evolves
snow, soil and groundwater storage through a fixed-step differentiable Heun
solver. Precipitation is partitioned into rain and snow, and the physical right
hand side includes melt, evapotranspiration, quickflow, percolation and
baseflow. At every derivative evaluation, routed RBF/Fourier or MLP experts
correct the five bounded flux-rate logits:

```text
dz/dt = physical water-balance RHS + expert-corrected flux terms
```

The storage derivatives sum exactly to `P - ET - Q`; test-time mass-balance
RMSE and negative-storage frequency are reported. Training carries the modeled
state chronologically between truncated-backpropagation chunks, matching the
single uninterrupted rollout used for validation and test. PINNMamba is
intentionally excluded from this experiment because
the present model is not an instantaneous vector field.

```bash
python -m examples.compare_hydrology_ude \
  --source camels_ch --basin 2011 --simple rbf --epochs 20
```

This compares analytic Morse routing, a jointly learned gate and the identical
physics-plus-MLP correction without routing. Fourier can replace RBF with
`--simple fourier`.
