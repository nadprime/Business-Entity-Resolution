"""Write requirements.txt with the exact versions installed in the CURRENT environment.
Run it in the environment where you did your final verification run."""
import argparse
import sys
from importlib import metadata

WANTED = ["numpy", "pandas", "scikit-learn", "xgboost", "rapidfuzz"]
FAISS_DISTS = ["faiss-gpu-cu12", "faiss-gpu-cu11", "faiss-gpu", "faiss-cpu", "faiss"]
OPTIONAL = ["cudf-cu12", "cupy-cuda12x"]


def version(dist):
    try:
        return metadata.version(dist)
    except metadata.PackageNotFoundError:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="requirements.txt")
    a = ap.parse_args()
    lines = [f"# Python {sys.version.split()[0]}"]
    for d in WANTED:
        v = version(d)
        lines.append(f"{d}=={v}" if v else f"# {d}: NOT INSTALLED")
    found = [(d, version(d)) for d in FAISS_DISTS if version(d)]
    lines += [f"{d}=={v}" for d, v in found] or ["# faiss: NOT INSTALLED"]
    lines.append("# optional (GPU ETL for the database build; falls back to pandas without them)")
    lines += [f"# {d}=={version(d)}" for d in OPTIONAL if version(d)]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
