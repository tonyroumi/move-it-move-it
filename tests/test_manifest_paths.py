from pathlib import Path

import pytest
import yaml

from moveitmoveit.manifest import load_manifest
from moveitmoveit.utils.paths import CONFIGS_DIR, MOTIONS_DIR

SHIPPED_MANIFESTS = sorted(CONFIGS_DIR.glob("*manifest*.yaml"))


def write_manifest(path: Path, file: str) -> Path:
    path.write_text(yaml.safe_dump({"motions": {"walk": {"file": file}}}))
    return path


def test_relative_file_resolves_against_data_dir(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.yaml", "jump/running_jump.npz")
    data_dir = tmp_path / "clips"

    (spec,) = load_manifest(str(manifest), data_dir=data_dir)

    assert spec.file == str(data_dir / "jump" / "running_jump.npz")


def test_relative_file_defaults_to_motions_dir(tmp_path):
    manifest = write_manifest(tmp_path / "manifest.yaml", "walk.npz")

    (spec,) = load_manifest(str(manifest))

    assert spec.file == str(MOTIONS_DIR / "walk.npz")


def test_absolute_file_is_kept(tmp_path):
    clip = tmp_path / "elsewhere" / "walk.npz"
    manifest = write_manifest(tmp_path / "manifest.yaml", str(clip))

    (spec,) = load_manifest(str(manifest), data_dir=tmp_path / "clips")

    assert spec.file == str(clip)


@pytest.mark.parametrize("manifest", SHIPPED_MANIFESTS, ids=lambda path: path.name)
def test_shipped_manifests_use_data_dir_relative_files(manifest):
    with open(manifest) as f:
        motions = yaml.safe_load(f)["motions"]

    for name, motion in motions.items():
        assert not Path(motion["file"]).is_absolute(), f"{manifest.name}:{name} hard-codes {motion['file']}"
    for spec in load_manifest(str(manifest), data_dir="/data"):
        assert spec.file.startswith("/data/")
