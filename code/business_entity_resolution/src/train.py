"""Train the XGBoost pair classifier and validate on a hold-out (nb1 cells 5-6).

Training data: for each of the first N_TRAIN Source-1 entities, take the FAISS
top-15 candidates plus every ground-truth match, label 1 if in ground truth else 0.
Hold-out: the next N_VAL Source-1 entities (never used for fitting).
"""
import json
import os
import sqlite3

import numpy as np
import pandas as pd
import xgboost as xgb

import blocking
import config
import matching
from features import extract_features
from preprocessing import load_source1


def load_ground_truth(path):
    gt = pd.read_csv(path, sep="\t", dtype=str)
    return {s1: (set(str(m).split(",")) if pd.notna(m) else set())
            for s1, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def entity_f05(true_set, pred_set):
    """Per-entity (precision, recall, F0.5). Entities with no true match score 1.0 only
    if nothing is predicted for them."""
    if not true_set:
        return (1.0, 1.0, 1.0) if not pred_set else (0.0, 1.0, 0.0)
    if not pred_set:
        return 1.0, 0.0, 0.0
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    denom = 0.25 * prec + rec
    return prec, rec, ((1.25 * prec * rec) / denom if denom > 0 else 0.0)


def macro_scores(entity_ids, gt_map, pred_map):
    scores = [entity_f05(gt_map.get(e, set()), pred_map.get(e, set())) for e in entity_ids]
    p, r, f = (float(np.mean([s[k] for s in scores])) for k in range(3))
    return {"precision": p, "recall": r, "f05": f}


def run_training(data_dir, db_path, model_out, n_train=config.N_TRAIN_ENTITIES,
                 n_val=config.N_VAL_ENTITIES, xgb_device="cuda", report_out=None):
    print("[train] loading Source-1 training split ...")
    s1_all = load_source1(os.path.join(data_dir, config.TRAIN_S1), nrows=n_train + n_val)
    s1_train = s1_all.iloc[:n_train].reset_index(drop=True)
    s1_val = s1_all.iloc[n_train:n_train + n_val].reset_index(drop=True)
    if len(s1_all) < n_train + n_val:
        print(f"[train] WARNING: only {len(s1_all):,} rows available "
              f"(requested {n_train + n_val:,}); hold-out has {len(s1_val):,} entities.")
    gt_map = load_ground_truth(os.path.join(data_dir, config.TRAIN_GT))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    countries = list(dict.fromkeys(list(s1_train["country"].unique())
                                   + list(s1_val["country"].unique())))
    print(f"[train] countries seen in train split: {sorted(s1_train['country'].unique())}")

    X_train, y_train = [], []
    val_pairs = {}   # country -> (query_df, cand_lists, pool) built once per country index

    for country in countries:
        # The full pool (train + test pool records) is indexed here, as in nb1. No test
        # labels exist or are used; test-pool records only act as unlabeled distractors.
        cidx = blocking.build_country_index(conn, country, valid_ids=None)

        s1_c = s1_train[s1_train["country"] == country].reset_index(drop=True)
        if len(s1_c):
            cands = blocking.query_candidates(cidx, s1_c["combined"].tolist(),
                                              config.TRAIN_TOP_K, config.TRAIN_MIN_SIM)
            all_ids = set()
            for i, s1_id in enumerate(s1_c["entity_id"]):
                all_ids.update(cands[i])
                all_ids.update(gt_map.get(s1_id, set()))
            pool = matching.bulk_fetch_candidates(cur, all_ids)
            for i, s1_id in enumerate(s1_c["entity_id"]):
                true_m = gt_map.get(s1_id, set())
                # sorted() only fixes row order for reproducibility; labels are unchanged.
                pair_ids = sorted(set(cands[i]) | true_m)
                for cid in pair_ids:
                    if cid in pool:
                        n2, a2, num2 = pool[cid]
                        X_train.append(extract_features(
                            s1_c.at[i, "name"], s1_c.at[i, "addr"], s1_c.at[i, "nums"],
                            n2, a2, num2))
                        y_train.append(1 if cid in true_m else 0)

        s1_v = s1_val[s1_val["country"] == country].reset_index(drop=True)
        if len(s1_v):
            v_cands = blocking.query_candidates(cidx, s1_v["combined"].tolist(),
                                                config.INFER_TOP_K, config.INFER_MIN_SIM)
            v_ids = {c for lst in v_cands for c in lst}
            val_pairs[country] = (s1_v, v_cands, matching.bulk_fetch_candidates(cur, v_ids))
        del cidx

    conn.close()
    X = np.asarray(X_train, dtype=np.float32)
    y = np.asarray(y_train, dtype=np.int64)
    print(f"[train] {len(y):,} training pairs ({int(y.sum()):,} positives). Fitting XGBoost ...")
    model = xgb.XGBClassifier(device=xgb_device, **config.XGB_PARAMS)
    model.fit(X, y)
    os.makedirs(os.path.dirname(os.path.abspath(model_out)), exist_ok=True)
    model.save_model(model_out)
    print(f"[train] model saved to {model_out}")

    report = evaluate_holdout(model, s1_val, val_pairs, gt_map)
    report.update({"n_train_entities": len(s1_train), "n_val_entities": len(s1_val),
                   "n_train_pairs": int(len(y)), "n_train_positives": int(y.sum()),
                   "train_countries": sorted(s1_train["country"].unique())})
    print(json.dumps(report, indent=2))
    if report_out:
        with open(report_out, "w") as fh:
            json.dump(report, fh, indent=2)
    return model


def evaluate_holdout(model, s1_val, val_pairs, gt_map):
    """Score the hold-out under (a) nb1's p>=0.75 rule and (b) the final submission rule.
    Also reports candidate (blocking) recall = share of true matches present in the candidates."""
    if len(s1_val) == 0:
        return {"note": "empty hold-out - nothing to evaluate"}
    preds = {"baseline_p>=0.75": {}, "final_rule": {}}
    found, total_true = 0, 0
    for country, (s1_v, cand_lists, pool) in val_pairs.items():
        X, owners, cand_ids = matching.build_pair_matrix(s1_v, cand_lists, pool)
        probs = matching.predict_proba(model, X)
        keep = {"baseline_p>=0.75": matching.decide_baseline(probs, X),
                "final_rule": matching.decide_final(probs, X)}
        for name, mask in keep.items():
            for e in s1_v["entity_id"]:
                preds[name][e] = set()
            for k in np.flatnonzero(mask):
                preds[name][s1_v.at[owners[k], "entity_id"]].add(cand_ids[k])
        for i, e in enumerate(s1_v["entity_id"]):
            true_set = gt_map.get(e, set())
            total_true += len(true_set)
            found += len(true_set & set(cand_lists[i]))
    ids = list(s1_val["entity_id"])
    report = {name: macro_scores(ids, gt_map, p) for name, p in preds.items()}
    report["candidate_recall"] = found / total_true if total_true else None
    report["candidate_settings"] = {"top_k": config.INFER_TOP_K, "min_sim": config.INFER_MIN_SIM}
    return report
