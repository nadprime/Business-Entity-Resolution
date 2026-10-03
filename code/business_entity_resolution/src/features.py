"""Pairwise similarity features (identical to nb1 / nb2 `extract_features`)."""
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

FEATURE_NAMES = ["Name Fuzz Ratio", "Name Token Sort", "Name Jaro-Winkler",
                 "Address Fuzz Ratio", "Address Numbers Jaccard", "Address Numbers Match"]
N_FEATURES = len(FEATURE_NAMES)
IDX_NAME_JW = 2        # used by the rescue rule
IDX_ADDR_RATIO = 3     # used by the rescue rule


def extract_features(n1, a1, num1_str, n2, a2, num2_str):
    """n*/a* = cleaned name/address, num*_str = comma-joined digit tokens of the address."""
    n1, a1, n2, a2 = n1 or "", a1 or "", n2 or "", a2 or ""
    nums1 = set(num1_str.split(",")) if num1_str else set()
    nums2 = set(num2_str.split(",")) if num2_str else set()
    nums1.discard("")
    nums2.discard("")
    jaccard = len(nums1 & nums2) / len(nums1 | nums2) if nums1 and nums2 else 0.0
    if nums1 and nums2 and (nums1 & nums2):
        num_match = 1.0
    elif not nums1 or not nums2:
        num_match = 0.5          # no number on one side: neutral, not evidence against
    else:
        num_match = 0.0
    return [fuzz.ratio(n1, n2) / 100.0,
            fuzz.token_sort_ratio(n1, n2) / 100.0,
            JaroWinkler.similarity(n1, n2),
            fuzz.ratio(a1, a2) / 100.0,
            jaccard,
            num_match]
