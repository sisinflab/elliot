# Changelog

All notable changes to this project will be documented in this file

## [v0.3.1] - 2021-07-05
### Changed
- fix AMF: insert of perturbation evaluation, insert multi-step adversarial perturbations
- fix AMR:  insert of perturbation evaluation, insert multi-step adversarial perturbations, improved the data pipeline
- start experiments with command line parameters (```python start_experiments.py --conf name_config_file```)
- released configuration files for DEMO paper (Under review at RecSys 2021)
- readme file in ```/data/``` directory with links to datasets

## [v0.3.0] - 2021-06-30
### Changed
- early stopping strategies
- offline recommendation files evaluation (ProxyRecommender, RecommendationFolder) 
- negative sampling evaluation
- improved Microsoft Windows compatibility  
- binarization of explicit dataset
- automatic loading of implicit datasets 
- multiple prefiltering
- managing side information with modular loaders
- alignment of side information with training data
- improved Documentation: Model creation, Side Information loading, Early Stopping, Negative Sampling 
- added nDCG as formulated in Rendle's 2020 KDD paper
- visual loader with tensorflow pipeline 
- added and fixing visual recsys method:
  - DVBPR
  - VBPR
  - DeepStyle
  - ACF
  - VNPR
- added new recommender method
  - MF (Rendle's 2020 RecSys reproducibility paper)
  - EASER
  - RP3beta
  - iALS

## [v0.2.1] - 2021-03-27
### Changed

- `requirements.txt` for Pillow vulnerabilities, change version to >=8.1.1
- Adversarial features for ECIR tutorial "__AL4Rec__"
- Hands-on example for ECIR Tutorial "__AL4Rec__"

## [v0.2.0] - 2021-03-12
### Added

- new dataloader ItemCategoryLoader
- Enabled FunSVD with batch_size
- Generalized feature-aware Factorization Machines
- batch_size field in recommendation snippet examples

### Changed

- MultiDAE documentation
- setup issues by [@deklanw](https://github.com/deklanw)
- indentation in README config by [@deklanw](https://github.com/deklanw)
- separated sample main and sample devel scripts
- paper experiment coherence
- combination Full recommendation metrics + argpartition
- Dataset data mock-ups
- long names namespace model generation during hyperspace exploration
- default validation k setup


## [v0.1.0] - 2021-03-05
First Release