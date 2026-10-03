"""Compare a regenerated matching_results.tsv with the file uploaded to the leaderboard.

Prints byte-identity (sha256) and, if they differ, how many entities / pairs differ
(row order is ignored, ids inside a row are compared as sets)."""
import argparse
import csv
import hashlib


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load(path):
    with open(path, newline="") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader)
        return {r[0]: set(r[1].split(",")) if len(r) > 1 and r[1] else set() for r in reader}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", required=True)
    ap.add_argument("--uploaded", required=True, help="the file you submitted to the leaderboard")
    a = ap.parse_args()
    ha, hb = sha256(a.generated), sha256(a.uploaded)
    print("sha256 generated:", ha, "\nsha256 uploaded :", hb)
    if ha == hb:
        print("IDENTICAL (byte-for-byte)")
        return
    g, u = load(a.generated), load(a.uploaded)
    only_g, only_u = set(g) - set(u), set(u) - set(g)
    diff = [k for k in g.keys() & u.keys() if g[k] != u[k]]
    pairs_g = sum(len(g[k] - u[k]) for k in g.keys() & u.keys())
    pairs_u = sum(len(u[k] - g[k]) for k in g.keys() & u.keys())
    print(f"NOT identical: {len(diff):,} entities differ out of {len(g.keys() & u.keys()):,}; "
          f"{pairs_g:,} pairs only in generated, {pairs_u:,} only in uploaded; "
          f"ids only in generated: {len(only_g)}, only in uploaded: {len(only_u)}")


if __name__ == "__main__":
    main()
