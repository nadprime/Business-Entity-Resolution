"""Text cleaning and the out-of-core SQLite pool database (nb1 cells 1-3)."""
import os
import re

import pandas as pd

import config

try:  # GPU ETL was used for the original database build; CPU fallback is below.
    import cudf
    USE_CUDF = True
except Exception:  # ImportError, or CUDA runtime missing
    cudf = None
    USE_CUDF = False

DB_COLUMNS = ["entity_id", "country", "name", "addr", "combined", "nums"]


def clean_text(text):
    """Lower-case, replace punctuation by space, collapse whitespace."""
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()


def extract_numbers(addr):
    """Comma-joined list of all stand-alone digit tokens in a cleaned address."""
    return ",".join(re.findall(r"\b\d+\b", addr))


def add_text_columns(df):
    """Pandas implementation of the cleaning. Adds country/name/addr/combined/nums."""
    df["country"] = df["country"].fillna("UNKNOWN").str.strip().str.upper()
    df["name"] = df["business_name"].apply(clean_text)
    df["addr"] = df["business_address"].apply(clean_text)
    df["combined"] = df["name"] + " " + df["addr"]
    df["nums"] = df["addr"].apply(extract_numbers)
    return df


def load_source1(path, nrows=None):
    """Load a Source-1 (query) file and add the cleaned columns."""
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=nrows)
    return add_text_columns(df)


def _clean_chunk_cudf(chunk_pd):
    gdf = cudf.from_pandas(chunk_pd)
    gdf["country"] = gdf["country"].fillna("UNKNOWN").str.strip().str.upper()
    for src, dst in (("business_name", "name"), ("business_address", "addr")):
        gdf[dst] = (gdf[src].fillna("").str.lower()
                    .str.replace(r"[^\w\s]", " ", regex=True)
                    .str.replace(r"\s+", " ", regex=True).str.strip())
    gdf["combined"] = gdf["name"] + " " + gdf["addr"]
    out = gdf.to_pandas()
    del gdf
    return out


def build_database(data_dir, db_path, chunksize=500_000):
    """Stream every pool file (Source 2 + 3, train AND test) into one SQLite table.

    The table keeps `country` indexed so the pool can be sliced per country.
    Built into a temporary file and renamed at the end, so an interrupted build
    can never be mistaken for a finished database.
    """
    import sqlite3

    if os.path.exists(db_path):
        print(f"[db] {db_path} already exists - skipping construction.")
        return
    tmp_path = db_path + ".building"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"[db] building {db_path} (cuDF ETL: {USE_CUDF})")
    conn = sqlite3.connect(tmp_path)
    cur = conn.cursor()
    cur.execute("PRAGMA synchronous = OFF")
    cur.execute("PRAGMA journal_mode = MEMORY")
    cur.execute("CREATE TABLE pool (entity_id TEXT PRIMARY KEY, country TEXT, "
                "name TEXT, addr TEXT, combined TEXT, nums TEXT)")

    for rel in config.POOL_FILES:
        path = os.path.join(data_dir, rel)
        print(f"[db] streaming {path}")
        for chunk in pd.read_csv(path, sep="\t", dtype=str, chunksize=chunksize):
            if USE_CUDF:
                chunk = _clean_chunk_cudf(chunk)
                chunk["nums"] = chunk["addr"].apply(extract_numbers)
            else:
                chunk = add_text_columns(chunk)
            # INSERT OR IGNORE: a duplicated entity_id keeps its first occurrence.
            cur.executemany("INSERT OR IGNORE INTO pool VALUES (?,?,?,?,?,?)",
                            chunk[DB_COLUMNS].values.tolist())

    cur.execute("CREATE INDEX idx_country ON pool(country)")
    conn.commit()
    conn.close()
    os.replace(tmp_path, db_path)
    print("[db] database built.")
