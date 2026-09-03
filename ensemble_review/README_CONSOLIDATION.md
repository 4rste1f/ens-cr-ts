# Isolated hydrology boundary-consolidation experiment

This experiment leaves the standard learned-Morse pipeline unchanged. It uses
the separate module `complexity_ensemble.hydrology_consolidation_comparison`
and entry point `examples.compare_learned_morse_hydrology_consolidation`.

The routed prediction remains a soft mixture. The optional loss is

```text
4 * w * (1 - w) * (simple_raw - complex_raw)^2
```

at each input window. It is strongest at the 50/50 routing boundary. Run the
isolated pipeline with consolidation enabled:

```bash
python -m examples.compare_learned_morse_hydrology_consolidation \
  --source camels_ch --basin 2011 --simple rbf --complex mlp \
  --epochs 750 --consolidation-weight 0.05 \
  --training-noise 0.1 --inference-noise 0.1
```

Use `--models` to select exactly which candidates to train. For example, this
trains only the single MLP baseline and Morse routing, without either learned
routing method or learned-Morse calibration:

```bash
python -m examples.compare_learned_morse_hydrology_consolidation \
  --source camels_ch --basin 2011 --complex mlp \
  --models single_complex,morse
```

The supported values are `morse`, `learned_morse`, `learned`, `tda`,
`lyapunov`, and `single_complex`. The learned-Morse potential is calibrated and assessed only
when `learned_morse` is selected. The comparison function already takes this
same explicit model list, so adding a future routing/estimation method requires
registering its model construction and then adding its name to the CLI choices.

The dynamical-complexity routes use a longer, past-only discharge history while
the experts retain their original `--sequence-length`. For example:

```bash
python -m examples.compare_learned_morse_hydrology_consolidation \
  --models morse,tda,lyapunov,single_complex \
  --sequence-length 30 --tda-context-length 128 \
  --lyapunov-context-length 365
```

`tda` constructs a Takens delay embedding and routes using an H1 persistence
summary. It requires `pip install -e '.[tda]'`. `lyapunov` uses a
confidence-filtered Rosenstein slope; estimates below `--lyapunov-min-r2` are
treated as invalid and receive zero complex weight. Both methods fit the soft
routing threshold using training scores only. Adjust the shared calibration
with `--complexity-percentile` and `--gate-temperature`. The comparison CSV
records mean test complexity and the valid-estimate fraction, and a JSON file
stores the complete CLI configuration.

For the otherwise identical ablation, use `--consolidation-weight 0`. Results
go to `artifacts/hydrology/learned_morse_consolidation` by default and include
the weight in filenames and the comparison CSV.

For each consolidation setting, the command trains both clean and noisy-input
models and evaluates both under clean and noisy inputs. Noise is Gaussian in
training-standardized units and is limited to precipitation, temperature, PET,
and previous discharge. The full 2x2 robustness grid is stored in the CSV.
The comparison PNG shows the same four conditions as grouped bars.
