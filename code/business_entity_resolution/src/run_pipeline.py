"""End-to-end pipeline: database -> (train) -> blocking+matching inference -> validation.
Uses the shipped model in models/xgb_entity_model.json. Add --retrain to train a new model from the training data.
"""
import argparse
import os
import sys

import config

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default="data",
                    help="folder with train/ and test/ sub-folders of official TSVs")
    ap.add_argument("--work_dir", default="work", help="scratch: SQLite db, retrained model, reports")
    ap.add_argument("--output_dir", default=os.path.abspath(os.path.join(ROOT, "..", "..", "output")))
    ap.add_argument("--model_path", default=os.path.join(ROOT, "models", config.DEFAULT_MODEL_NAME))
    ap.add_argument("--retrain", action="store_true", help="train a new model instead of using --model_path")
    ap.add_argument("--xgb_device", default="cuda", help="cuda or cpu (training only)")
    ap.add_argument("--n_train", type=int, default=config.N_TRAIN_ENTITIES, help="training entities (retrain)")
    ap.add_argument("--n_val", type=int, default=config.N_VAL_ENTITIES, help="hold-out entities (retrain)")
    ap.add_argument("--stage", default="all", choices=["all", "db", "train", "predict", "validate"])
    a = ap.parse_args()

    os.makedirs(a.work_dir, exist_ok=True)
    db_path = os.path.join(a.work_dir, "pool_data.db")
    model_path = a.model_path
    stages = ["db", "train", "predict", "validate"] if a.stage == "all" else [a.stage]
    if a.stage == "train":
        a.retrain = True          # asking for the train stage explicitly implies retraining

    if set(stages) & {"db", "train", "predict"}:
        from preprocessing import build_database
        build_database(a.data_dir, db_path)   # no-op if the database already exists

    if a.retrain:
        model_path = os.path.join(a.work_dir, "xgb_entity_model_retrained.json")
        if "train" in stages:
            from train import run_training
            run_training(a.data_dir, db_path, model_path, n_train=a.n_train, n_val=a.n_val,
                         xgb_device=a.xgb_device,
                         report_out=os.path.join(a.work_dir, "validation_report.json"))

    if "predict" in stages:
        from predict import run_inference
        run_inference(a.data_dir, db_path, model_path, a.output_dir)

    if "validate" in stages:
        from validate_outputs import validate
        issues, stats = validate(os.path.join(a.output_dir, "matching_results.tsv"),
                                 os.path.join(a.output_dir, "candidate_pairs.tsv"), a.data_dir)
        print("stats:", stats)
        if issues:
            print(f"VALIDATION FAILED ({len(issues)} issues):", *issues[:10], sep="\n - ")
            sys.exit(1)
        print("VALIDATION PASSED")


if __name__ == "__main__":
    main()
