"""Pair construction, XGBoost scoring and the decision rules."""
import numpy as np

import config
from features import IDX_ADDR_RATIO, IDX_NAME_JW, N_FEATURES, extract_features


def bulk_fetch_candidates(cur, cand_ids):
    """entity_id -> (name, addr, nums) for the given ids (SQLite variable limit safe)."""
    pool = {}
    ids = list(set(cand_ids))
    for i in range(0, len(ids), 900):
        chunk = ids[i:i + 900]
        cur.execute("SELECT entity_id, name, addr, nums FROM pool WHERE entity_id IN "
                    f"({','.join(['?'] * len(chunk))})", chunk)
        for eid, name, addr, nums in cur.fetchall():
            pool[eid] = (name, addr, nums)
    return pool


def build_pair_matrix(query_df, cand_lists, pool):
    """Flatten (query, candidate) pairs into one feature matrix.

    Returns X (n_pairs x 6), owners (query row index per pair) and cand_ids (per pair).
    Candidates missing from `pool` are skipped, as in the notebooks.
    """
    names, addrs, nums = (query_df["name"].tolist(), query_df["addr"].tolist(),
                          query_df["nums"].tolist())
    rows, owners, cand_ids = [], [], []
    for i, cands in enumerate(cand_lists):
        for cid in cands:
            rec = pool.get(cid)
            if rec is None:
                continue
            rows.append(extract_features(names[i], addrs[i], nums[i], *rec))
            owners.append(i)
            cand_ids.append(cid)
    X = np.asarray(rows, dtype=np.float64).reshape(-1, N_FEATURES)
    return X, np.asarray(owners, dtype=np.int64), cand_ids


def predict_proba(model, X):
    if len(X) == 0:
        return np.zeros(0, dtype=np.float32)
    return model.predict_proba(X)[:, 1]


def decide_final(probs, X):
    """Final submission rule: p>=0.93, or p>=0.85 with near-perfect name & address strings."""
    strict = probs >= config.XGB_STRICT_THRESH
    rescue = ((probs >= config.XGB_RESCUE_THRESH)
              & (X[:, IDX_NAME_JW] >= config.RESCUE_NAME_JW)
              & (X[:, IDX_ADDR_RATIO] >= config.RESCUE_ADDR_RATIO))
    return strict | rescue


def decide_baseline(probs, X=None):
    """nb1's plain threshold (p >= 0.75)."""
    return probs >= config.BASELINE_THRESH
