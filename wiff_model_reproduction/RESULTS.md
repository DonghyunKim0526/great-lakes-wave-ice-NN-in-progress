# Executed WIFF1.0 component reproduction

## Dataset and numerical results

The official archive contains **5,087,436 states**. This run selects **100,000 rows uniformly without replacement**, seed **20261024**. All 100,000 selected rows pass the paper's thickness, concentration, and wave-height screen; 52,237 have nonzero reference fracture histograms.

| Diagnostic | Measured result | Interpretation |
| --- | ---: | --- |
| Classifier accuracy | 87.636% | Archived pretrained weights; paper reports about 87.5% |
| False-positive rate | 10.661% | False calls divided by reference nonfracture states |
| False-negative rate | 13.921% | Missed calls divided by reference fracture states |
| Mean representative size error | 29.374 m | Weighted absolute histogram error; paper gives about 30 m |
| Median RSE / R | 12.789% | Paper Eq. 6-9 definition, computed on archived fracture states |
| Median symmetric mean-radius error | 3.904% | Released plotting-helper definition; numerically close to reported 3.9% |
| Normalized NN histograms | 100.0% | All evaluated probability vectors sum to one |

**This is an archived-data diagnostic, not an independent held-out score.** Original split membership is not available, and the pretrained weights may have been trained on sampled rows. Numerical proximity to a paper statistic does not establish reproduction of the original validation split.

## Main finding: the two error definitions are not interchangeable

The paper defines `RSE = sum(r_i * abs(q_pred_i - q_true_i))` and `SSE = RSE / R_true`. The released `Paper_Figures/Misc/training_preamble.m` instead defines `perc_error = 100 * abs(R_true - R_pred) / (0.5 * (R_true + R_pred))`. The latter allows errors in different bins to cancel when forming the mean radius; the former does not.

On exactly the same 52,237 states, the medians are **12.79%** and **3.90%**, respectively. We preserve both definitions and do not relabel the smaller number as weighted histogram error. This source-code/equation discrepancy is material to interpreting a reproduction of the reported 3.9% result. The [metric-comparison figure](figures/07_metric_definition_comparison.png) makes it visible. This finding concerns these source definitions and this run; it is not a claim that the original held-out outputs have been independently audited.

## Fresh numerical kernel comparisons

- Randomly selected physical states: **24**; numerical stopping criterion met in **24**.
- Median NN-versus-fresh-SP weighted histogram error: **30.39%**.
- Median fixed-phase single-iteration-versus-fresh-SP weighted histogram error: **18.99%**.
- Median Python SP kernel latency: **38.755 ms/state**.
- Median warmed NN histogram latency: **0.0615 ms/state**.
- Ratio of those median component latencies: **630.7x**.

These 24 states and newly generated phase realizations do not establish generalization to new climates or observations. The SP kernel is the documented MATLAB stand-alone port, not a recompiled CICE Fortran kernel. In this small comparison, the error values above must be retained even if the neural model does not outperform a one-iteration reference. The speed ratio is a component benchmark on this computer, not an end-to-end CICE speedup. Fourier-basis construction, disk loading, classification, I/O, and wave-model execution are excluded; one BLAS thread is used.

## Reproducibility and evidence

- [metrics.json](results/metrics.json): full numerical results and execution environment.
- [source_provenance.json](results/source_provenance.json): official data/code URLs, checksums, and sizes.
- [archive_diagnostics.csv](results/archive_diagnostics.csv): all sampled row IDs and classifier outcomes; row IDs are zero-based in the archive.
- [fresh_sp_cases.csv](results/fresh_sp_cases.csv): all selected cases, timing, stopping status, and errors.
- [fresh_sp_histograms.npz](results/fresh_sp_histograms.npz): selected inputs, archived labels, new numerical histograms, and NN outputs.
- [convergence_example.json](results/convergence_example.json): exact published stand-alone example input and five seeded stopping histories.
- `python -m pytest -q`: tests of text/HDF5 weight parity, row-wise softmax, literal MATLAB peak/curvature logic, physical metrics, seeds, and upstream file checksums.

The raw 753 MB data remain outside Git tracking. No Great Lakes adaptation, new network training, global CICE rerun, or paper Figures 4-5 reproduction is claimed.
