"""Candidate generation (blocking): hashed char-3-gram -> 64-d random projection
-> exact cosine nearest-neighbour search with FAISS, one index per country."""
from functools import lru_cache

import faiss
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

import config

_VECTORIZER = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3),
                                n_features=config.HASH_FEATURES, dtype=np.float32)


@lru_cache(maxsize=1)
def _projection():
    # RandomState(42).randn(...) yields exactly the same numbers as the notebooks'
    # np.random.seed(42); np.random.randn(...), without touching global RNG state.
    rng = np.random.RandomState(config.PROJECTION_SEED)
    mat = rng.randn(config.HASH_FEATURES, config.VECTOR_DIM).astype(np.float32)
    return mat / np.sqrt(config.VECTOR_DIM)


def text_to_dense_l2(texts):
    """Strings -> sparse hashed 3-grams -> dense 64-d -> L2-normalised (cosine via inner product)."""
    sparse = _VECTORIZER.transform(texts)
    dense = np.ascontiguousarray(sparse.dot(_projection()).astype(np.float32))
    faiss.normalize_L2(dense)
    return dense


class CountryIndex:
    """Exact inner-product FAISS index (GPU if available, else CPU) + row -> entity_id map."""

    def __init__(self):
        cpu_index = faiss.IndexFlatIP(config.VECTOR_DIM)
        self._gpu_resources = None  # must outlive the GPU index
        n_gpus = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        if n_gpus > 0 and hasattr(faiss, "StandardGpuResources"):
            res = faiss.StandardGpuResources()
            res.noTempMemory()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            self._gpu_resources = res
        else:
            self.index = cpu_index
        self.ids = []

    def __len__(self):
        return len(self.ids)


def build_country_index(conn, country, valid_ids=None):
    """Stream one country's pool records from SQLite into a FAISS index.

    valid_ids=None  -> index every pool record (train + test pool); used for training.
    valid_ids=set   -> index only records whose id is in the set; used for test
                       inference so that only TEST pool entities can be returned.
    """
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM pool WHERE country=?", (country,))
    total = cur.fetchone()[0]
    cidx = CountryIndex()
    mode = "test-only" if valid_ids is not None else "full pool"
    print(f"[faiss] building {mode} index for {country} ({total:,} pool records scanned)")

    for offset in range(0, total, config.INDEX_CHUNK_SIZE):
        cur.execute("SELECT entity_id, combined FROM pool WHERE country=? LIMIT ? OFFSET ?",
                    (country, config.INDEX_CHUNK_SIZE, offset))
        rows = cur.fetchall()
        if not rows:
            break
        if valid_ids is not None:
            rows = [r for r in rows if r[0] in valid_ids]
            if not rows:
                continue
        cidx.ids.extend(r[0] for r in rows)
        cidx.index.add(text_to_dense_l2([r[1] for r in rows]))
    print(f"[faiss]   indexed {len(cidx):,} records")
    return cidx


def query_candidates(cidx, texts, top_k, min_sim):
    """Top-k cosine neighbours per query text, dropping hits below min_sim.

    Returns a list (one entry per query) of entity_id lists ordered best-first.
    """
    out = []
    if len(cidx) == 0:
        return [[] for _ in texts]
    for start in range(0, len(texts), config.QUERY_BATCH_SIZE):
        batch = texts[start:start + config.QUERY_BATCH_SIZE]
        sims, idxs = cidx.index.search(text_to_dense_l2(batch), top_k)
        for row_sims, row_idxs in zip(sims, idxs):
            out.append([cidx.ids[j] for j, s in zip(row_idxs, row_sims)
                        if j != -1 and s >= min_sim])
    return out
