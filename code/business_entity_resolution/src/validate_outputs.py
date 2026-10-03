"""Format + consistency check of the two output TSVs, and summary statistics for the
write-up (candidate-pair count etc.). Exit code 0 = PASS, 1 = FAIL.

Checks: exact headers; one row per test Source-1 id (no missing/extra/duplicate);
every candidate and every match id exists in the TEST pool (Source 2/3); no duplicate ids
inside a row; every match is also a candidate for that entity.
"""
import argparse
import csv
import os
import sys

import pandas as pd

import config

MATCH_HEADER = ["source1_entity_id", "matched_entity_ids"]
CAND_HEADER = ["source1_entity_id", "candidate_entity_ids"]


def _ids(path):
    return set(pd.read_csv(path, sep="\t", usecols=["entity_id"], dtype=str)["entity_id"].dropna())


def _rows(path, header, issues, name):
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        if next(reader, None) != header:
            issues.append(f"{name}: bad header")
        for row in reader:
            if len(row) != 2:
                issues.append(f"{name}: malformed row {row[:1]}")
                continue
            yield row[0], (row[1].split(",") if row[1] else [])


def validate(matching_file, candidate_file, test_dir_or_data_dir):
    base = test_dir_or_data_dir
    test_dir = os.path.join(base, "test") if os.path.isdir(os.path.join(base, "test")) else base
    valid_s1 = _ids(os.path.join(test_dir, "test_source1.tsv"))
    valid_pool = _ids(os.path.join(test_dir, "test_source2.tsv")) | _ids(
        os.path.join(test_dir, "test_source3.tsv"))

    issues = []
    stats = {"entities": 0, "candidate_pairs": 0, "matched_pairs": 0,
             "entities_with_match": 0, "entities_with_no_candidate": 0}

    cand_map, seen = {}, set()
    for s1, cands in _rows(candidate_file, CAND_HEADER, issues, "candidates"):
        if s1 in seen:
            issues.append(f"candidates: duplicate source1 id {s1}")
        seen.add(s1)
        if len(cands) != len(set(cands)):
            issues.append(f"candidates: duplicate ids in {s1}")
        bad = [c for c in cands if c not in valid_pool]
        if bad:
            issues.append(f"candidates: {len(bad)} id(s) not in test pool for {s1}, e.g. {bad[0]}")
        cand_map[s1] = set(cands)
        stats["candidate_pairs"] += len(cands)
        stats["entities_with_no_candidate"] += not cands
    if seen != valid_s1:
        issues.append(f"candidates: source1 ids differ from test_source1 "
                      f"(missing {len(valid_s1 - seen)}, extra {len(seen - valid_s1)})")

    seen = set()
    for s1, matches in _rows(matching_file, MATCH_HEADER, issues, "matches"):
        if s1 in seen:
            issues.append(f"matches: duplicate source1 id {s1}")
        seen.add(s1)
        if len(matches) != len(set(matches)):
            issues.append(f"matches: duplicate ids in {s1}")
        for m in matches:
            if m not in valid_pool:
                issues.append(f"matches: invalid id {m} in {s1}")
            if m not in cand_map.get(s1, ()):
                issues.append(f"matches: {m} not among candidates of {s1}")
        stats["entities"] += 1
        stats["matched_pairs"] += len(matches)
        stats["entities_with_match"] += bool(matches)
    if seen != valid_s1:
        issues.append(f"matches: source1 ids differ from test_source1 "
                      f"(missing {len(valid_s1 - seen)}, extra {len(seen - valid_s1)})")
    return issues, stats


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matching", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--data_dir", required=True, help="folder containing test/ (or the test folder itself)")
    a = ap.parse_args()
    issues, stats = validate(a.matching, a.candidate, a.data_dir)
    print("stats:", stats)
    if stats["entities"]:
        print(f"avg candidates / entity: {stats['candidate_pairs'] / stats['entities']:.2f}")
    if issues:
        print(f"FAIL ({len(issues)} issue(s)); first 10:")
        for i in issues[:10]:
            print(" -", i)
        sys.exit(1)
    print("PASS: format and consistency checks OK")


if __name__ == "__main__":
    main()
