# WIFF1.0 Reproduction and Great Lakes Adaptation

**Status: planned.** This directory defines the next experimental stage. It does not yet contain a completed reproduction, trained model, or benchmark results.

## Reference

Horvat and Roach (2022), WIFF1.0: [published article](https://gmd.copernicus.org/articles/15/803/2022/).

See the [original paper](../literature%20review/5.ML-wave-ice-parametrization/original.pdf) and [existing summary](../literature%20review/5.ML-wave-ice-parametrization/summary.pdf).

## Reproduction plan

1. Identify and pin the authors' code and data releases; record versions, licenses, dependencies, and execution instructions.
2. Verify the original input/output contract, including spectral bins, units, preprocessing, ice thickness and concentration, and floe size categories.
3. Run the authors' baseline implementation on its intended data and compare outputs with reported results before any regional adaptation.
4. Assess Great Lakes wave and ice data coverage, spectral compatibility, and validation options.
5. If feasible, construct an adaptation dataset, evaluate a surrogate, and document accuracy, physical consistency, and runtime against the chosen numerical reference.

## Working data requirements

| Quantity | Initial expectation | Unresolved requirement |
| --- | --- | --- |
| Wave spectrum | 25 binned energy values | Frequency bins, units, integration, and conversion from candidate wave products |
| Ice thickness | One input value per sample | Source, spatial/temporal alignment, and uncertainty |
| Ice concentration | One input value per sample | Units, preprocessing, and alignment |
| Floe size distribution | 12 output categories | Category definitions, normalization, numerical targets, and observational validation |

These dimensions reflect the current literature review and planning notes; they are not an implemented dataset specification.

## Evaluation design to finalize

- Define train/validation/test splits that limit leakage across time, location, and related model states.
- Compare distribution predictions and relevant physical diagnostics with the numerical reference.
- Record inference and reference-model runtime under comparable hardware and batch sizes; report preprocessing costs separately.
- Check valid output ranges, distribution constraints, edge cases, and behavior outside the training domain.

## Open questions

- Which Great Lakes products provide the required spectra, thickness, and concentration together?
- Are numerical reference outputs and floe size observations available for validation?
- Can the required ice-model coupling be run in the Great Lakes configuration?
- Which assumptions in the original model require regional reassessment?
