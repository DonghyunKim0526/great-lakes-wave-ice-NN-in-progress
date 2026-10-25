# Great Lakes Wave–Ice Dynamics

### Spectral Modeling and Machine Learning

This research investigates wave–ice interactions in the Great Lakes by connecting spectral wave modeling, coupled wave–ice physics, and scientific machine learning. The central question is whether neural network surrogates can represent selected wave attenuation or ice fracture processes accurately while reducing computational cost relative to numerical models.

**Current stage:** literature review. WIFF1.0 reproduction, surrogate development, and numerical benchmarking are planned; no reproduction or model performance results are claimed in this repository.

## Research objectives

- Synthesize the physical assumptions and parameterizations used in spectral wave and coupled wave–ice models.
- Identify the inputs, outputs, observations, and numerical reference data needed for Great Lakes applications.
- Assess the feasibility of reproducing WIFF1.0 before adapting its approach to Great Lakes conditions.
- Subject to data availability and feasibility, develop a neural network surrogate and compare its predictive accuracy and computational efficiency with a numerical reference model.

## Repository structure

```text
README.md
literature review/
├── 1.review/
├── 2.one_way_coupled/
├── 3.wavewatch3_icewave/
├── 4.fully-coupled/
└── 5.ML-wave-ice-parametrization/
    # Every category contains original.pdf and summary.pdf.
wiff-reproduction/
└── README.md
```

## Literature review

| Theme | Focus | Original | Summary |
| --- | --- | --- | --- |
| Review | Physical mechanisms, observations, and wave–ice modeling context | [PDF](literature%20review/1.review/original.pdf) | [PDF](literature%20review/1.review/summary.pdf) |
| One-way coupling | Ice attenuation and Great Lakes numerical modeling | [PDF](literature%20review/2.one_way_coupled/original.pdf) | [PDF](literature%20review/2.one_way_coupled/summary.pdf) |
| WAVEWATCH III ice–wave modeling | Spectral modeling and ice parameterizations | [PDF](literature%20review/3.wavewatch3_icewave/original.pdf) | [PDF](literature%20review/3.wavewatch3_icewave/summary.pdf) |
| Fully coupled modeling | Feedbacks between wave and ice states | [PDF](literature%20review/4.fully-coupled/original.pdf) | [PDF](literature%20review/4.fully-coupled/summary.pdf) |
| ML wave–ice parameterization | WIFF1.0 and neural network representations of floe fracture | [PDF](literature%20review/5.ML-wave-ice-parametrization/original.pdf) | [PDF](literature%20review/5.ML-wave-ice-parametrization/summary.pdf) |

Original PDFs are source publications; summaries are existing Korean-language research notes. Files were collected from the local research workspace and its Overleaf material and retained without altering their contents. Historical labels inside the notes reflect their original preparation context. Source publications retain their respective authorship and rights.

## WIFF1.0 reproduction and Great Lakes adaptation

The next research step is [WIFF1.0 Reproduction and Great Lakes Adaptation](wiff-reproduction/README.md), based on Horvat and Roach (2022). Initial planning considers 25 binned wave spectral inputs, ice thickness, and ice concentration, with a 12-category floe size distribution as the target output. Exact binning, units, normalization, and output interpretation must be verified against the paper and implementation before experiments begin.

Candidate sources include NOAA GLERL wave and ice products. Availability of suitable wave spectra, ice thickness, and floe size observations remains to be established. Compatibility of ice models such as CICE with the proposed Great Lakes workflow is an open feasibility question.

## Planned outputs

1. A literature synthesis of spectral wave models and machine learning approaches to wave–ice interactions.
2. A documented WIFF1.0 reproduction feasibility assessment and, if feasible, reproducible experiments.
3. A prototype neural network surrogate, conditional on suitable data and project scope.
4. An evaluation of predictive performance and computational cost against an existing numerical model once reference data are available.
5. A research report and presentation describing findings, limitations, and future directions.

## Resources

- [Horvat and Roach (2022), WIFF1.0](https://gmd.copernicus.org/articles/15/803/2022/)
- [NOAA GLERL ice information](https://www.glerl.noaa.gov/data/ice/#overview)

