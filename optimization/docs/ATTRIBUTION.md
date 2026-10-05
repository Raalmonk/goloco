# Attribution and licenses

This is an independent performance and reproducibility contribution to [pritchardlabatpsu/goloco](https://github.com/pritchardlabatpsu/goloco). It is not an official Pritchard-lab release. The original authors are credited for GOLOCO and its scientific methods; they are not represented as contributors to, reviewers of, or endorsers of this optimization.

## Original application and scientific work

**GOLOCO application:** Sajid M. Hossain, Yiyun Rao, Jahid O. Hossain, Justin R. Pritchard, and Boyang Zhao. *goloco: a web application to create genome scale information from surprisingly small experiments.* BMC Bioinformatics **26**, 61 (2025), published 25 February 2025. [DOI: 10.1186/s12859-025-06070-y](https://doi.org/10.1186/s12859-025-06070-y). Author order and publication metadata were verified against the [publisher's article](https://link.springer.com/article/10.1186/s12859-025-06070-y).

**Foundational lossy-compression methods:** Boyang Zhao, Yiyun Rao, Scott Leighow, Edward P. O’Brien, Luke Gilbert, and Justin R. Pritchard. *A pan-CRISPR analysis of mammalian cell specificity identifies ultra-compact sgRNA subsets for genome-scale experiments.* Nature Communications **13**, 625 (2022), published 2 February 2022. [DOI: 10.1038/s41467-022-28045-w](https://doi.org/10.1038/s41467-022-28045-w). Metadata was verified against the [publisher's article](https://www.nature.com/articles/s41467-022-28045-w).

The tested upstream source is commit [`470477c4be6ba0095016a2089b9bbf33d49a1ef8`](https://github.com/pritchardlabatpsu/goloco/tree/470477c4be6ba0095016a2089b9bbf33d49a1ef8). GOLOCO's [archived software record](https://doi.org/10.5281/zenodo.14249073) is separate from the inference-model dataset.

## Released model dataset

The official model record is **Sajid Hossain, MD. *ceres-infer dataset for goloco webapp* (2024)**, published 30 November 2024, [DOI: 10.5281/zenodo.14251390](https://doi.org/10.5281/zenodo.14251390). The record identifies the creator as “Sajid Hossain, MD”; this citation retains that published form.

The [official Zenodo record](https://zenodo.org/records/14251390) and its [metadata API](https://zenodo.org/api/records/14251390), checked during publication preparation, identify:

| Field | Published value |
| --- | --- |
| File | `ceres-infer.zip` |
| File size | 5,130,807,234 bytes |
| MD5 | `3811e49729e22d3146d2d57c04f0834c` |
| Record license | Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)) |

This contribution publishes the converter, checksums, compact benchmark evidence, and independently generated performance figures. It does not redistribute the author model archive, source model pickles, or the converted native bundle. See [data availability](DATA_AVAILABILITY.md) and [reproduction instructions](REPRODUCE.md) for retrieval and conversion. The retained source estimators remain necessary for the accepted backend's legacy input validation.

## Preserved notices and scope

- The upstream [MIT license](../../LICENSE), copyright 2023 PritchardlabatPSU, is preserved unchanged. The additive optimization code is distributed under that repository license.
- The embedded Chronos [license notice](../../chronos/LICENSE), copyright 2021, 2023 Joshua M. Dempster, is preserved unchanged. Chronos fitting is outside this optimization's measured scope.
- The application article is published under CC BY-NC-ND 4.0. This contribution cites the article and does not reproduce its figures or article text. The new figures visualize the recorded performance measurements in this contribution.
- Third-party Python libraries and the Rust toolchain remain separately licensed dependencies. Their binaries, environments, font files, and model artifacts are not included here. The Rust crate has no external crate dependencies.

The code, evidence audit, and publication checks were developed with automated coding assistance. Historical model-dependent checks are identified as historical experiment evidence; fresh publication checks cover source preservation, evidence processing, figures, repository hygiene, and links. No human peer review, upstream acceptance, or scientific improvement in prediction accuracy is claimed.
