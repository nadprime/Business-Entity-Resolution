# Trained model

`xgb_entity_model.json` is the XGBoost pair classifier used to produce the submitted
`output/matching_results.tsv`.

* Produced by the training step of the pipeline (`src/train.py`; originally
  `notebooks/nb1_training_and_first_inference.ipynb`, cell 5, saved in cell 8).
* Hyper-parameters: 450 trees, learning rate 0.05, max depth 6, `hist` tree method,
  logloss, `random_state=42`, trained on a GPU.
* Trained on 40,000 Source-1 entities from `train_source1.tsv` (first 40,000 rows)
  with FAISS top-15 candidates + all ground-truth matches. No pretrained weights,
  no external data.
* 6 input features, in this order: name fuzz ratio, name token-sort ratio, name
  Jaro-Winkler, address fuzz ratio, address-number Jaccard, address-number match.

`run_pipeline.py` loads this file by default. Use `--retrain` to train a fresh model instead.
