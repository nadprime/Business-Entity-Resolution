"""Central configuration. Every value here is taken from the original notebooks
(nb1 = training, nb2 = final inference) so the packaged pipeline reproduces them."""

# ---- Blocking (identical in nb1 and nb2) -----------------------------------
HASH_FEATURES = 2 ** 15        # char 3-gram hashing space
VECTOR_DIM = 64                # random-projection size (nb1 final cell: 64)
PROJECTION_SEED = 42           # seed of the Gaussian projection matrix
INDEX_CHUNK_SIZE = 200_000     # rows streamed from SQLite per FAISS add()
QUERY_BATCH_SIZE = 5_000       # queries per FAISS search() call

# Candidate retrieval used to BUILD TRAINING DATA (nb1, cell 5)
TRAIN_TOP_K = 15
TRAIN_MIN_SIM = 0.25

# Candidate retrieval used for FINAL TEST INFERENCE (nb2, cell 5)
INFER_TOP_K = 20
INFER_MIN_SIM = 0.20

# ---- Matching ---------------------------------------------------------------
# Decision rule used in the final submission (nb2)
XGB_STRICT_THRESH = 0.93       # accept if p >= 0.93
XGB_RESCUE_THRESH = 0.85       # ...or if p >= 0.85 AND name JW >= 0.90 AND addr ratio >= 0.85
RESCUE_NAME_JW = 0.90
RESCUE_ADDR_RATIO = 0.85

# Decision rule used in nb1's validation (reported F0.5 = 0.8344)
BASELINE_THRESH = 0.75

# XGBoost hyper-parameters (nb1, cell 5). `device` is chosen at run time.
XGB_PARAMS = dict(
    n_estimators=450,
    learning_rate=0.05,
    max_depth=6,
    tree_method="hist",
    eval_metric="logloss",
    random_state=42,
)

# Training / validation split of train_source1 (nb1: first 50k rows)
N_TRAIN_ENTITIES = 40_000
N_VAL_ENTITIES = 10_000

# ---- Expected data layout under --data_dir ---------------------------------
TRAIN_S1 = "train/train_source1.tsv"
TRAIN_GT = "train/train_ground_truth.tsv"
POOL_FILES = [
    "train/train_source2.tsv",
    "train/train_source3.tsv",
    "test/test_source2.tsv",
    "test/test_source3.tsv",
]
TEST_S1 = "test/test_source1.tsv"
TEST_S2 = "test/test_source2.tsv"
TEST_S3 = "test/test_source3.tsv"

DEFAULT_MODEL_NAME = "xgb_entity_model.json"
