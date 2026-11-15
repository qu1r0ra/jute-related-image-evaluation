import hashlib
import json
from pathlib import Path

import pytest

from jute_disease.cli import main
from jute_disease.data import (
    SCENE_SET_ALLOCATION,
    draw_scene_sets,
    read_candidates,
    read_scene_set,
)
from tests.data.frozen_targets import Member, content_id, singles, write_target

POOLS = ("A", "B", "C")
SMALL = {pool: {"held_out": 3, "reserve_fit": 2, "reserve_test": 2} for pool in POOLS}
SET_FILES = ("held-out.json", "reserve-fit.json", "reserve-test.json")


def _members(pool: str, train: int = 14, val: int = 6) -> list[Member]:
    return [
        *singles(pool, "train", train),
        *singles(pool, "val", val),
        *singles(pool, "test", 3),
    ]


def _world(
    tmp_path: Path,
    members: dict[str, list[Member]] | None = None,
    quarantined: dict[str, tuple[str, ...]] | None = None,
    development: tuple[str, ...] = (),
) -> dict:
    members = members or {}
    targets = {
        pool: write_target(
            tmp_path / "targets" / pool,
            pool,
            members.get(pool, _members(pool)),
            quarantined=(quarantined or {}).get(pool, ()),
            grouping_version="dhash-v3" if pool == "C" else "dhash-v2",
            threshold=7 if pool == "C" else 6,
        )
        for pool in POOLS
    }
    dev = tmp_path / "development.json"
    dev.write_text(
        json.dumps(
            [
                {
                    "image_id": content_id(name),
                    "path": f"{name}.png",
                    "scene_tag": "field",
                }
                for name in development
            ]
        ),
        encoding="utf-8",
    )
    return {
        "targets": targets,
        "image_roots": {pool: tmp_path / "images" / pool for pool in POOLS},
        "development": dev,
    }


def _draw(world: dict, output: Path, **options) -> dict:
    options.setdefault("seed", 7)
    options.setdefault("allocation", SMALL)
    return draw_scene_sets(
        world["targets"],
        world["image_roots"],
        world["development"],
        output,
        **options,
    )


def _ids(output: Path) -> set[str]:
    drawn = {
        record["image_id"]
        for name in SET_FILES
        for record in read_scene_set(output / name)
    }
    inventory = json.loads((output / "audit-inventory.json").read_text("utf-8"))
    return drawn | {record["image_id"] for record in inventory}


def test_default_allocation_is_the_protocol_one() -> None:
    assert {
        pool: sizes["held_out"] for pool, sizes in SCENE_SET_ALLOCATION.items()
    } == {
        "A": 60,
        "B": 60,
        "C": 180,
    }
    assert sum(sizes["reserve_fit"] for sizes in SCENE_SET_ALLOCATION.values()) == 170
    assert sum(sizes["reserve_test"] for sizes in SCENE_SET_ALLOCATION.values()) == 100
    assert SCENE_SET_ALLOCATION["C"] == {
        "held_out": 180,
        "reserve_fit": 102,
        "reserve_test": 60,
    }


def test_draw_writes_sized_sets_from_train_and_val_only(tmp_path: Path) -> None:
    world = _world(tmp_path)
    output = tmp_path / "sets"

    receipt = _draw(world, output)

    for name, key in zip(
        SET_FILES, ("held_out", "reserve_fit", "reserve_test"), strict=True
    ):
        records = read_scene_set(output / name)
        for pool in POOLS:
            chosen = [record for record in records if record["pool"] == pool]
            assert len(chosen) == SMALL[pool][key] == receipt["counts"][pool][key]
            assert {record["partition"] for record in chosen} <= {"train", "val"}
            assert all(
                Path(record["path"]).is_relative_to(world["image_roots"][pool])
                for record in chosen
            )
    inventory = read_scene_inventory(output)
    assert len(inventory) == 3 * (20 - 7)
    assert all(set(record) == {"image_id", "path"} for record in inventory)
    assert receipt["counts"]["A"]["audit_inventory"] == 13
    assert receipt["seed"] == 7
    assert receipt["grouping"]["C"] == {
        "version": "dhash-v3",
        "threshold": 7,
        "digest": json.loads(
            (world["targets"]["C"] / "grouping.json").read_text("utf-8")
        )["digest"],
    }
    assert receipt["grouping"]["A"]["version"] == "dhash-v2"
    assert len(receipt["manifest_digests"]["A"]["fit"]) == 64
    assert len(receipt["manifest_digests"]["A"]["protected_test"]) == 64
    assert (
        receipt["development_manifest_digest"]
        == hashlib.sha256(world["development"].read_bytes()).hexdigest()
    )
    assert json.loads((output / "receipt.json").read_text("utf-8")) == receipt


def read_scene_inventory(output: Path) -> list[dict]:
    return json.loads((output / "audit-inventory.json").read_text("utf-8"))


def test_the_inventory_is_readable_as_mask_audit_candidates(tmp_path: Path) -> None:
    world = _world(tmp_path)
    output = tmp_path / "sets"
    _draw(world, output)
    for record in read_scene_inventory(output):
        path = Path(record["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not opened")

    candidates = read_candidates(output / "audit-inventory.json")

    assert len(candidates) == 39


def test_same_inputs_and_seed_give_the_same_output(tmp_path: Path) -> None:
    world = _world(tmp_path)

    first = _draw(world, tmp_path / "one")
    second = _draw(world, tmp_path / "two")
    other = _draw(world, tmp_path / "three", seed=8)

    for name in (*SET_FILES, "audit-inventory.json"):
        assert (tmp_path / "one" / name).read_bytes() == (
            tmp_path / "two" / name
        ).read_bytes()
    assert first["outputs"] == second["outputs"]
    assert first["outputs"] != other["outputs"]


def test_outputs_are_disjoint_by_image_and_family(tmp_path: Path) -> None:
    members = {
        pool: [
            *(
                Member(f"{pool}-train-{index}", f"{pool}-fam-{index // 2}", "train")
                for index in range(24)
            ),
            *singles(pool, "val", 6),
            *singles(pool, "test", 3),
        ]
        for pool in POOLS
    }
    world = _world(tmp_path, members)
    output = tmp_path / "sets"
    family = {
        content_id(member.name): (pool, member.group)
        for pool, pool_members in members.items()
        for member in pool_members
    }

    receipt = _draw(world, output)

    sets = [
        {record["image_id"] for record in read_scene_set(output / name)}
        for name in SET_FILES
    ]
    sets.append({record["image_id"] for record in read_scene_inventory(output)})
    for index, left in enumerate(sets):
        for right in sets[index + 1 :]:
            assert not left & right
            assert not {family[i] for i in left} & {family[i] for i in right}
    unused = sum(receipt["exclusions"][pool]["unused_family_members"] for pool in POOLS)
    assert sum(map(len, sets)) + unused == 3 * 30


def test_quarantined_and_test_images_are_never_eligible(tmp_path: Path) -> None:
    world = _world(tmp_path, quarantined={"A": ("A-quarantined-0",)})
    output = tmp_path / "sets"

    _draw(world, output)

    ids = _ids(output)
    assert content_id("A-quarantined-0") not in ids
    assert not ids & {
        content_id(f"{pool}-test-{i}") for pool in POOLS for i in range(3)
    }
    assert len(ids) == 60


def test_a_quarantined_fit_image_is_excluded(tmp_path: Path) -> None:
    world = _world(tmp_path, quarantined={"A": ("A-train-0",)})
    output = tmp_path / "sets"

    receipt = _draw(world, output)

    assert content_id("A-train-0") not in _ids(output)
    assert receipt["exclusions"]["A"]["quarantined"] == 1


def test_an_image_in_more_than_one_pool_is_excluded_from_each(tmp_path: Path) -> None:
    members = {
        "A": [*_members("A"), Member("shared-fit", "A-shared", "train")],
        "B": [
            *_members("B"),
            Member("shared-fit", "B-shared", "val"),
            Member("shared-with-test", "B-shared-2", "train"),
        ],
        "C": [*_members("C"), Member("shared-with-test", "C-shared", "test")],
    }
    world = _world(tmp_path, members)
    output = tmp_path / "sets"

    receipt = _draw(world, output)

    assert not _ids(output) & {content_id("shared-fit"), content_id("shared-with-test")}
    assert receipt["exclusions"]["A"]["cross_pool_repeat"] == 1
    assert receipt["exclusions"]["B"]["cross_pool_repeat"] == 2
    assert receipt["exclusions"]["C"]["cross_pool_repeat"] == 0


def test_development_and_explicitly_excluded_families_are_excluded(
    tmp_path: Path,
) -> None:
    members = {
        "A": [
            *_members("A"),
            Member("dev-image", "A-dev-family", "train"),
            Member("dev-sibling", "A-dev-family", "train"),
            Member("ruled-out", "A-ruled-family", "val"),
            Member("ruled-sibling", "A-ruled-family", "val"),
        ]
    }
    world = _world(tmp_path, members, development=("dev-image",))
    output = tmp_path / "sets"

    receipt = _draw(world, output, excluded_ids=[content_id("ruled-out")])

    names = ("dev-image", "dev-sibling", "ruled-out", "ruled-sibling")
    assert not _ids(output) & {content_id(name) for name in names}
    assert receipt["exclusions"]["A"]["development_family"] == 2
    assert receipt["exclusions"]["A"]["excluded_family"] == 2
    assert receipt["development_images"] == 1
    assert receipt["excluded_ids"] == 1


def test_a_development_image_no_manifest_knows_is_refused(tmp_path: Path) -> None:
    world = _world(tmp_path)
    world["development"].write_text(
        json.dumps(
            [{"image_id": "dev-1", "path": "missing.png", "scene_tag": "field"}]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sets"

    with pytest.raises(ValueError, match="1 development image.*dev-1"):
        _draw(world, output)

    assert not output.exists()


def test_an_excluded_id_no_manifest_knows_is_refused(tmp_path: Path) -> None:
    world = _world(tmp_path)
    output = tmp_path / "sets"

    with pytest.raises(ValueError, match="1 excluded image.*not-an-image"):
        _draw(world, output, excluded_ids=["not-an-image"])

    assert not output.exists()


def test_a_development_image_is_matched_by_file_content(tmp_path: Path) -> None:
    world = _world(tmp_path)
    image = tmp_path / "dev-copy.png"
    image.write_bytes(b"development image bytes")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    target = world["targets"]["B"]
    path = target / "manifest.json"
    data = json.loads(path.read_text("utf-8"))
    data["entries"][0]["image_id"] = digest
    data["entries"].sort(key=lambda entry: entry["image_id"])
    from jute_disease.data.manifest import Manifest, write_manifest

    write_manifest(Manifest.from_json(data), target)
    world["development"].write_text(
        json.dumps(
            [{"image_id": "dev-1", "path": "dev-copy.png", "scene_tag": "field"}]
        ),
        encoding="utf-8",
    )
    output = tmp_path / "sets"

    receipt = _draw(world, output)

    assert digest not in _ids(output)
    assert receipt["exclusions"]["B"]["development_family"] == 1


def test_a_family_with_a_protected_test_member_is_excluded(tmp_path: Path) -> None:
    world = _world(tmp_path)
    target = world["targets"]["A"]
    path = target / "protected-test" / "manifest.json"
    data = json.loads(path.read_text("utf-8"))
    data["entries"][0]["group"] = "A-train-g0"
    from jute_disease.data.manifest import Manifest, write_manifest

    write_manifest(Manifest.from_json(data), target / "protected-test")
    output = tmp_path / "sets"

    receipt = _draw(world, output)

    assert content_id("A-train-0") not in _ids(output)
    assert receipt["exclusions"]["A"]["protected_test_family"] == 1


def test_too_few_eligible_families_blocks_the_draw(tmp_path: Path) -> None:
    members = {
        "B": [
            *(Member(f"B-train-{i}", f"B-fam-{i // 5}", "train") for i in range(20)),
            Member("B-val-0", "B-fam-val", "val"),
            *singles("B", "test", 3),
        ]
    }
    world = _world(tmp_path, members)

    with pytest.raises(ValueError, match="pool B has too few eligible families"):
        _draw(world, tmp_path / "sets")

    assert not (tmp_path / "sets").exists()


def test_existing_output_is_never_overwritten(tmp_path: Path) -> None:
    world = _world(tmp_path)
    output = tmp_path / "sets"
    output.mkdir()

    with pytest.raises(FileExistsError):
        _draw(world, output)

    assert not list(output.iterdir())


def test_a_failed_write_leaves_nothing_behind(tmp_path: Path, monkeypatch) -> None:
    world = _world(tmp_path)
    output = tmp_path / "out" / "sets"
    original = Path.write_text

    def failing(self: Path, *args: object, **kwargs: object) -> int:
        if self.name == "reserve-fit.json":
            raise OSError("disk full")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", failing)
    with pytest.raises(OSError, match="disk full"):
        _draw(world, output)
    monkeypatch.undo()

    assert not list(output.parent.iterdir())
    _draw(world, output)
    assert (output / "reserve-fit.json").is_file()


def test_pools_must_match_the_allocation(tmp_path: Path) -> None:
    world = _world(tmp_path)
    del world["targets"]["C"]

    with pytest.raises(ValueError, match="pools"):
        _draw(world, tmp_path / "sets")


def test_scene_sets_command_draws_the_protocol_sizes(tmp_path: Path, capsys) -> None:
    sizes = {"A": (90, 30), "B": (90, 30), "C": (270, 80)}
    members = {pool: _members(pool, *sizes[pool]) for pool in POOLS}
    world = _world(tmp_path, members, development=("A-train-0",))
    excluded = tmp_path / "excluded.json"
    excluded.write_text(json.dumps([content_id("B-train-0")]), encoding="utf-8")
    output = tmp_path / "sets"
    arguments = [
        "data",
        "scene-sets",
        "--development",
        str(world["development"]),
        "--exclude",
        str(excluded),
        "--seed",
        "11",
    ]
    for pool in POOLS:
        arguments += ["--target-dir", f"{pool}={world['targets'][pool]}"]
        arguments += ["--image-root", f"{pool}={world['image_roots'][pool]}"]

    main([*arguments, "--output", str(output)])

    report = json.loads(capsys.readouterr().out)
    assert report["counts"]["C"] == {
        "eligible": 350,
        "held_out": 180,
        "reserve_fit": 102,
        "reserve_test": 60,
        "audit_inventory": 8,
    }
    assert len(read_scene_set(output / "held-out.json")) == 300
    assert len(read_scene_set(output / "reserve-fit.json")) == 170
    assert len(read_scene_set(output / "reserve-test.json")) == 100
    assert report["exclusions"]["A"]["development_family"] == 1
    assert report["exclusions"]["B"]["excluded_family"] == 1
    with pytest.raises(SystemExit, match="scene sets blocked"):
        main([*arguments, "--output", str(output)])
