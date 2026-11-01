from jute_disease.data import dhash_candidates


def test_pigeonhole_index_finds_all_pairs_within_four_bits() -> None:
    hashes = {"zero": 0, "four": 0x1000100010001, "far": 0xFFFFFFFFFFFFFFFF, "one": 1}
    assert dhash_candidates(hashes) == [
        ("four", "one", 3),
        ("four", "zero", 4),
        ("one", "zero", 1),
    ]
