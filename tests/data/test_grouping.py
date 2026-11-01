from jute_disease.data import dhash_candidates


def test_pigeonhole_index_finds_all_pairs_within_four_bits() -> None:
    hashes = {"zero": 0, "four": 0x1000100010001, "far": 0xFFFFFFFFFFFFFFFF, "one": 1}
    assert dhash_candidates(hashes) == [
        ("four", "one", 3),
        ("four", "zero", 4),
        ("one", "zero", 1),
    ]


def test_flip_canonical_dhash() -> None:
    from PIL import Image, ImageOps

    from jute_disease.data import flip_canonical_dhash

    image = Image.new("L", (9, 8))
    image.putdata([0, 255, 0, 255, 0, 255, 0, 255, 0] * 8)
    assert flip_canonical_dhash(image) == 0x5555555555555555
    assert flip_canonical_dhash(ImageOps.mirror(image)) == 0x5555555555555555


def test_components_and_audit_include_chains_and_cross_label_edges() -> None:
    from jute_disease.data import audit_components, union_find_components

    hashes = {"a": 0, "b": 0xFFFF, "c": 0xFFFFFFFF}
    edges = [("a", "b", 16), ("b", "c", 16)]
    components = union_find_components(hashes, edges)
    assert components == [("a", "b", "c")]
    audit = audit_components(hashes, {"a": "x", "b": "x", "c": "y"}, edges)
    assert audit[0]["max_distance"] == 32
    assert audit[0]["cross_label_bridges"] == [("b", "c")]
    assert audit[0]["flags"] == ["distance", "cross_label"]


def test_repair_removes_rated_edges_in_distance_order_and_reaudits() -> None:
    from jute_disease.data import repair_components

    hashes = {"a": 0, "b": 1, "c": 3, "d": 15}
    edges = [("a", "b", 1), ("b", "c", 1), ("a", "d", 4)]
    result = repair_components(
        hashes, dict.fromkeys(hashes, "x"), edges, {("a", "b"), ("a", "d")}
    )
    assert result["removed"] == [("a", "d", 4), ("a", "b", 1)]
    assert [a["members"] for a in result["audit"]] == [("a",), ("b", "c"), ("d",)]


def test_source_screen_compares_sha_and_hash_against_all_pools() -> None:
    from jute_disease.data import screen_source_overlap

    source = {
        "s1": ("sha1", 0),
        "s2": ("sha2", 0xFFFFFFFFFFFFFFFF),
        "s3": ("same", 123),
    }
    targets = {"A": {"t1": ("different", 1)}, "B": {"t2": ("same", 0)}}
    assert screen_source_overlap(source, targets) == {
        "s1": [("A", "t1", "dhash"), ("B", "t2", "dhash")],
        "s3": [("B", "t2", "sha256")],
    }


def test_grouping_drops_entire_conflicting_component_after_audit() -> None:
    from jute_disease.data import group_inventory

    images = {
        "a": ("sha-a", 0, "x"),
        "b": ("sha-b", 1, "y"),
        "c": ("sha-c", 0xFFFFFFFFFFFFFFFF, "x"),
    }
    result = group_inventory(images)
    assert result["audit"][0]["flags"] == ["cross_label"]
    assert result["excluded"] == ["a", "b"]
    assert result["families"] == [("c",)]


def test_exact_conflicting_copies_cannot_be_repaired_apart() -> None:
    from jute_disease.data import group_inventory

    result = group_inventory(
        {"a": ("same", 0, "x"), "b": ("same", 1, "y")}, {("a", "b")}
    )
    assert result["excluded"] == ["a", "b"]
