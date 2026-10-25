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
├── 5.ML-wave-ice-parametrization/
└── translation/
    ├── 1.review/
    ├── 2.one_way_coupled/
    ├── 3.wavewatch3_icewave/
    ├── 4.fully-coupled/
    └── 5.ML-wave-ice-parametrization/
    # Every category contains a paper PDF named by title and author, plus summary.pdf.
wiff_model_reproduction/
└── README.md
```

## Literature review

| Theme | Focus | Original | English summary |
| --- | --- | --- | --- |
| Review | Physical mechanisms, observations, and wave–ice modeling context | [PDF](literature%20review/1.review/Ocean%20Wave%20Interactions%20with%20Sea%20Ice%20-%20A%20Reappraisal%20-%20Vernon%20A.%20Squire%20%282020%29.pdf) | [PDF](literature%20review/1.review/summary.pdf) |
| One-way coupling | Ice attenuation and Great Lakes numerical modeling | [PDF](literature%20review/2.one_way_coupled/Modeling%20the%20ice-attenuated%20waves%20in%20the%20Great%20Lakes%20-%20Peng%20Bai%20et%20al.%20%282020%29.pdf) | [PDF](literature%20review/2.one_way_coupled/summary.pdf) |
| WAVEWATCH III ice–wave modeling | Spectral modeling and ice parameterizations | [PDF](literature%20review/3.wavewatch3_icewave/Winter%20Ice-Wave%20Modeling%20With%20WAVEWATCH%20III%20in%20Lake%20Erie%20-%20Haoguo%20Hu%20et%20al.%20%282025%29.pdf) | [PDF](literature%20review/3.wavewatch3_icewave/summary.pdf) |
| Fully coupled modeling | Feedbacks between wave and ice states | [PDF](literature%20review/4.fully-coupled/Simulating%20ice-wave%20interactions%20in%20the%20Laurentian%20Great%20Lakes%20using%20a%20fully%20coupled%20hydrodynamic-ice-wave%20model%20-%20M.%20Javad%20Javaherian%20et%20al.%20%282025%29.pdf) | [PDF](literature%20review/4.fully-coupled/summary.pdf) |
| ML wave–ice parameterization | WIFF1.0 and neural network representations of floe fracture | [PDF](literature%20review/5.ML-wave-ice-parametrization/WIFF1.0%20-%20a%20hybrid%20machine-learning-based%20parameterization%20of%20wave-induced%20sea%20ice%20floe%20fracture%20-%20Horvat%20and%20Roach%20%282022%29.pdf) | [PDF](literature%20review/5.ML-wave-ice-parametrization/summary.pdf) |

Original PDFs are source publications. Each `summary.pdf` in the five literature categories is a newly written English synthesis based directly on the corresponding original paper, with research questions, methods, key results, limitations, and implications for the Great Lakes project. Each includes every numbered figure from its source publication, with figure-by-figure reading notes and source attribution (5, 7, 10, 18, and 5 figures, respectively; 45 in total). These English summaries are separate from the Korean notes preserved under `translation/`.

Source materials were collected from the local research workspace and its Overleaf material. Reader-added highlights, handwriting, and note annotations have been removed from the paper PDFs; publication page content and links are preserved. Paper filenames include the title, author, and year. Source publications and reproduced figures retain their respective authorship and rights.

## Korean translations

The [translation directory](literature%20review/translation) mirrors the five literature categories. Each folder contains the existing Korean paper translation named by paper title, author, and year, with a `Korean Translation` suffix, and the preserved Korean research notes as `summary.pdf`. These Korean notes are independent of the new English summaries in the parent literature categories and were not changed when the English summaries were added.

| Theme | Translation | Korean summary |
| --- | --- | --- |
| Review | [PDF](literature%20review/translation/1.review/Ocean%20Wave%20Interactions%20with%20Sea%20Ice%20-%20A%20Reappraisal%20-%20Vernon%20A.%20Squire%20%282020%29%20-%20Korean%20Translation.pdf) | [PDF](literature%20review/translation/1.review/summary.pdf) |
| One-way coupling | [PDF](literature%20review/translation/2.one_way_coupled/Modeling%20the%20ice-attenuated%20waves%20in%20the%20Great%20Lakes%20-%20Peng%20Bai%20et%20al.%20%282020%29%20-%20Korean%20Translation.pdf) | [PDF](literature%20review/translation/2.one_way_coupled/summary.pdf) |
| WAVEWATCH III ice–wave modeling | [PDF](literature%20review/translation/3.wavewatch3_icewave/Winter%20Ice-Wave%20Modeling%20With%20WAVEWATCH%20III%20in%20Lake%20Erie%20-%20Haoguo%20Hu%20et%20al.%20%282025%29%20-%20Korean%20Translation.pdf) | [PDF](literature%20review/translation/3.wavewatch3_icewave/summary.pdf) |
| Fully coupled modeling | [PDF](literature%20review/translation/4.fully-coupled/Simulating%20ice-wave%20interactions%20in%20the%20Laurentian%20Great%20Lakes%20using%20a%20fully%20coupled%20hydrodynamic-ice-wave%20model%20-%20M.%20Javad%20Javaherian%20et%20al.%20%282025%29%20-%20Korean%20Translation.pdf) | [PDF](literature%20review/translation/4.fully-coupled/summary.pdf) |
| ML wave–ice parameterization | [PDF](literature%20review/translation/5.ML-wave-ice-parametrization/WIFF1.0%20-%20a%20hybrid%20machine-learning-based%20parameterization%20of%20wave-induced%20sea%20ice%20floe%20fracture%20-%20Horvat%20and%20Roach%20%282022%29%20-%20Korean%20Translation.pdf) | [PDF](literature%20review/translation/5.ML-wave-ice-parametrization/summary.pdf) |

## WIFF1.0 reproduction and Great Lakes adaptation

The next research step is [WIFF1.0 Reproduction and Great Lakes Adaptation](wiff_model_reproduction/README.md), based on Horvat and Roach (2022). Initial planning considers 25 binned wave spectral inputs, ice thickness, and ice concentration, with a 12-category floe size distribution as the target output. Exact binning, units, normalization, and output interpretation must be verified against the paper and implementation before experiments begin.

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

