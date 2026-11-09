# Elliot – Agent Guide

Elliot is a **configuration-driven recommender-systems evaluation framework**. A single YAML file drives the entire pipeline: data loading → splitting → hyperparameter search → training → evaluation → results.

## Running Experiments

```bash
# CLI (config name without .yaml extension):
python start_experiments.py --config test_config

# Python API:
from elliot.run import run_experiment
run_experiment("config_files/test_config.yaml")
```

## Pipeline Architecture

```
YAML config
  └─► build_namespace()          # elliot/namespace/ – YAML → Pydantic v2 models
        └─► DataSetLoader.build()    # elliot/dataset/ – load, prefilter, split
              └─► for each model:
                    run_hyperopt() / run_single()   # elliot/hyperoptimization/
                      └─► ModelCoordinator → AbstractTrainer → model.train_step()
                            └─► Evaluator.eval()   # elliot/evaluation/
  └─► ResultHandler              # elliot/result_handler/ – aggregate & write reports
```

Key entry point: `elliot/run.py::run_experiment()`.

## Config File Essentials

Every config must include `version: 0.3.1` (checked against `__version__` in `run.py`; mismatch raises an exception).

Hyperparameter search spaces are expressed inline per-model:
```yaml
models:
  BPRMF:
    meta:
      hyper_opt_alg: tpe        # tpe | grid | rand | anneal | atpe | mix
      hyper_max_evals: 20
      validation_metric: nDCG@10
    factors: [8, 16, 32]            # list = discrete choices (grid or TPE)
    learning_rate: [loguniform, -10, -2]   # hyperopt distribution
    lambda_user: [quniform, 0, 0.01, 0.001]
```

Reference configs: `config_files/test_config.yaml` (minimal), `config_files/advanced_configuration.yaml` (cross-val, complex metrics, WandB).

## Adding a New Recommender Model

1. **Create the model class** under `elliot/recommender/<category>/my_model.py`.
   - Inherit from `Recommender` (standard) or `GraphBasedRecommender` (graph-based), both in `elliot/recommender/base_recommender.py`.
   - Declare hyperparameters as **class-level annotated attributes** – they are auto-wired from the config by `set_params()`:
     ```python
     class MyModel(Recommender):
         factors: int = 64
         learning_rate: float = 0.001
     ```
   - Implement `train_step(self, batch, *args) -> float/Tensor` (required abstract method).
   - Optionally override `get_training_dataloader(batch_size)`.
   - Set `self.params_to_save` to list attributes that should be persisted as weights.

2. **Register** the model in `elliot/recommender/__init__.py`.

3. **External models** (e.g. specialised GNN/multimodal) live in `external/models/`, are registered in `external/models/__init__.py`, and referenced in YAML as `external.ModelName`. Backend-conditional imports use `config.backend` (`"tensorflow"` or `"pytorch"`).

## Trainer Pattern

`AbstractTrainer` (`elliot/recommender/base_trainer.py`) handles the full training loop, validation scheduling, early stopping, and result logging. Custom trainers can be added alongside a model class; `get_trainer()` in `elliot/utils/utils.py` resolves which trainer to use via module introspection.

## Device / GPU Selection

- `gpu: 0` in the YAML sets `CUDA_VISIBLE_DEVICES`.
- `device: mps|cpu|cuda` explicitly selects the torch device.
- Environment variable `ELLIOT_DEVICE=cuda|mps|cpu` is also honoured.
- Auto-detection order: CUDA → MPS → CPU (`elliot/utils/utils.py::_auto_device()`).

## Data Format

Interaction files are tab-separated: `user_id\titem_id\trating`. The placeholder `{0}` in `dataset_path` is replaced with the `dataset` field value:
```yaml
data_config:
  dataset_path: ../data/{0}/dataset.tsv
```
Datasets live in `data/<dataset_name>/`.

## Evaluation & Metrics

Metrics listed under `evaluation.simple_metrics` are computed at every cutoff in `evaluation.cutoffs`. The `validation_metric` (e.g. `nDCG@10`) **must** be present in `simple_metrics` and its cutoff **must** be in `cutoffs`; violations raise exceptions at startup.

The evaluation protocol lives in `evaluation.protocol` and is shared by every model of the experiment:
- `target: all|first|last|random|successive` decides what each eval row predicts (default: `last` if any model is SESSION_ONLY, `all` otherwise; validation uses `last` unless `all`). SESSION_ONLY models allow only `last|random|successive`; SESSION_ONLY mixed with other models allows only `last`, with `test_splitting: {strategy: temporal_holdout, leave_n_out: 1}`. Rules live in `ExperimentConfig.validate_evaluation_target()`.
- `negative_sampling` (optional) only decides the candidates (targets + sampled negatives per user); absent means full ranking.
- `mask_seen` (default `true`) excludes already-seen items from full-ranking recommendations.
- `user_filter: {path, id_space: public|private, reader}` (optional) restricts evaluation to the users listed in `path`.

Fields set at their old location (`evaluation.target`, top-level `negative_sampling`, …) raise an error pointing to the new one.

The protocol lives in `elliot/evaluation/protocol/`, not in `DataSet` (which only holds a fold's train/eval data):
- `EvalProtocol` (one per fold) builds and caches the eval rows (`EvalTargets`, ground truth and user filter applied) per session strategy, and samples the negatives (`NegativeSampler`) once per fold, so every model of the fold is scored on the same rows and negatives. `prepare_protocols(config, val_data, main_data)`, called once in `run.py` right after `DataSetLoader.build()`, attaches it to each fold's `DataSet` as `dataset.protocol` (the dataset only hosts it and never imports `evaluation`; reading `protocol` before preparation raises a `RuntimeError`). A test fold standing in for missing validation folds is prepared once.
- `EvalProtocol.get_eval_dataloader(batch_size, session_strategy)` wraps the cached `NegEvalDataset`/`FullEvalDataset` of that strategy in a dataloader (its dataset carries its rows, `dataloader.dataset.rows`).
- At every evaluation the trainer requests that dataloader from `dataset.protocol` (with its eval batch size and the model's session strategy) and `collect(model, dataloader, protocol, k)` (`elliot/evaluation/protocol/collector.py`) scores only its rows, returning a `CollectedRecs` (recs, ground truth, row→owner map) that `Evaluator.eval()` consumes as is. `SequentialRecommender`s also receive each row's context sequence (`user_seq`, `seq_len`), built by the collector from the eval rows.

## Testing

```bash
pytest tests/
```

`tests/params.py` provides `generate_param_combinations()` for building parametrised test cases. Config validation tests use `config_test: True` in the YAML to run dry without actually training.

## WandB Integration

Use one of the ready-made configs (`config_files/wandb_online.yaml`, `wandb_offline.yaml`, `wandb_disabled.yaml`) or add a `wandb:` block. Mode is resolved in `elliot/run.py::_setup_wandb()`.

