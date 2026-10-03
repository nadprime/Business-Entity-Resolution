"""Final test inference (nb2): TEST-ONLY FAISS candidates + XGBoost + final decision rule.
Writes candidate_pairs.tsv and matching_results.tsv."""
import csv
import gc
import os
import sqlite3

import pandas as pd
import xgboost as xgb

import blocking
import config
import matching
from preprocessing import load_source1


def load_test_pool_ids(data_dir):
    ids = set()
    for rel in (config.TEST_S2, config.TEST_S3):
        df = pd.read_csv(os.path.join(data_dir, rel), sep="\t", usecols=["entity_id"], dtype=str)
        ids.update(df["entity_id"].dropna())
    return ids


def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            "Copy the trained xgb_entity_model.json to models/ or run with --retrain.")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    model.set_params(device="cpu")   # inference on CPU, as in nb2
    return model


def run_inference(data_dir, db_path, model_path, output_dir, chunk_size=5000):
    os.makedirs(output_dir, exist_ok=True)
    cand_path = os.path.join(output_dir, "candidate_pairs.tsv")
    match_path = os.path.join(output_dir, "matching_results.tsv")

    print("[predict] loading test pool ids (only these may be returned) ...")
    valid_ids = load_test_pool_ids(data_dir)
    model = load_model(model_path)
    s1_test = load_source1(os.path.join(data_dir, config.TEST_S1))
    print(f"[predict] {len(s1_test):,} Source-1 test entities, {len(valid_ids):,} test pool ids")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    n_batches = 0
    with open(cand_path, "w", newline="") as f_cand, open(match_path, "w", newline="") as f_match:
        cw = csv.writer(f_cand, delimiter="\t")
        mw = csv.writer(f_match, delimiter="\t")
        cw.writerow(["source1_entity_id", "candidate_entity_ids"])
        mw.writerow(["source1_entity_id", "matched_entity_ids"])

        for country in s1_test["country"].unique():
            s1_c = s1_test[s1_test["country"] == country].reset_index(drop=True)
            cidx = blocking.build_country_index(conn, country, valid_ids=valid_ids)
            print(f"[predict] {country}: {len(s1_c):,} entities")

            for start in range(0, len(s1_c), chunk_size):
                n_batches += 1
                chunk = s1_c.iloc[start:start + chunk_size].reset_index(drop=True)
                cand_lists = blocking.query_candidates(
                    cidx, chunk["combined"].tolist(), config.INFER_TOP_K, config.INFER_MIN_SIM)
                pool = matching.bulk_fetch_candidates(
                    cur, {c for lst in cand_lists for c in lst})

                # One batched predict per chunk (row-wise identical to per-entity calls).
                X, owners, pair_ids = matching.build_pair_matrix(chunk, cand_lists, pool)
                probs = matching.predict_proba(model, X)
                accept = matching.decide_final(probs, X)
                matches = [[] for _ in range(len(chunk))]
                for k in range(len(pair_ids)):
                    if accept[k]:
                        matches[owners[k]].append(pair_ids[k])

                for i, s1_id in enumerate(chunk["entity_id"]):
                    cw.writerow([s1_id, ",".join(cand_lists[i])])
                    mw.writerow([s1_id, ",".join(matches[i])])
                print(f"  -> batch {n_batches}: {start + len(chunk):,}/{len(s1_c):,} ({country})")
                gc.collect()
            del cidx
            gc.collect()
    conn.close()
    print(f"[predict] wrote {cand_path}\n[predict] wrote {match_path}")
