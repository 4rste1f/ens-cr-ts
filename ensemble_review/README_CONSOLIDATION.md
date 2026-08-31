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

For the otherwise identical ablation, use `--consolidation-weight 0`. Results
go to `artifacts/hydrology/learned_morse_consolidation` by default and include
the weight in filenames and the comparison CSV.

For each consolidation setting, the command trains both clean and noisy-input
models and evaluates both under clean and noisy inputs. Noise is Gaussian in
training-standardized units and is limited to precipitation, temperature, PET,
and previous discharge. The full 2x2 robustness grid is stored in the CSV.
The comparison PNG shows the same four conditions as grouped bars.
