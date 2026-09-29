"""Vendored data: pinned copies of the Tallowbrook dataset and the RAG retrieval snapshot.

Each vendored directory has a MANIFEST.json with the source repo, the commit
the files came from and the sha256 of every file. `verify` checks the files
against it. `copy_from` refreshes a directory from a source checkout and
refuses one that is not at the pinned commit or has uncommitted changes, so a
manifest always names a commit the files really came from.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


@dataclass(frozen=True)
class VendoredSource:
    name: str
    dest: Path
    source_repo: str
    pinned_commit: str
    files: dict[str, str]  # source path -> vendored file name
    license: str
    description: str


TALLOWBROOK = VendoredSource(
    name="tallowbrook",
    dest=DATA_DIR / "tallowbrook",
    source_repo=(
        "rkemery/rag-support-assistant, tag tallowbrook-v0.1 (branch claude/tallowbrook-dataset)"
    ),
    pinned_commit="3c72058e8cc7d5dd6e224ecdec7b814341a3d48f",
    files={
        "agents/tasks.jsonl": "tasks.jsonl",
        "agents/bank_seed.json": "bank_seed.json",
        "facts/policies.yaml": "policies.yaml",
    },
    license="CC-BY-4.0",
    description="Tallowbrook Neobank Support (synthetic): agent tasks, fake bank seed, facts file",
)

RAG_SNAPSHOT = VendoredSource(
    name="rag_snapshot",
    dest=DATA_DIR / "rag_snapshot",
    source_repo="rag-support-assistant",
    pinned_commit="88a3b5ce56e9253240750b6663b4cfb745c80551",
    files={
        "snapshot/chunks.jsonl": "chunks.jsonl",
        "snapshot/embeddings.npy": "embeddings.npy",
        "snapshot/config.json": "config.json",
        "snapshot/README.md": "SNAPSHOT_README.md",
    },
    license="CC-BY-4.0 (chunk text from the Tallowbrook dataset)",
    description="Frozen retrieval snapshot of the RAG help-center index (bge-small + BM25)",
)

SOURCES = (TALLOWBROOK, RAG_SNAPSHOT)


class VendorError(RuntimeError):
    """Vendored files do not match their manifest, or the source is not the pinned commit."""


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(source: VendoredSource) -> list[str]:
    """Check every vendored file against the manifest. Returns the verified file names."""
    manifest_path = source.dest / "MANIFEST.json"
    if not manifest_path.exists():
        raise VendorError(f"missing {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("source_commit") != source.pinned_commit:
        raise VendorError(
            f"{manifest_path} names commit {manifest.get('source_commit')}, "
            f"expected {source.pinned_commit}"
        )
    problems = []
    for entry in manifest["files"]:
        path = source.dest / entry["path"]
        if not path.exists():
            problems.append(f"missing {entry['path']}")
        elif sha256(path) != entry["sha256"]:
            problems.append(f"hash mismatch for {entry['path']}")
    listed = {entry["path"] for entry in manifest["files"]}
    if listed != set(source.files.values()):
        problems.append(
            f"manifest lists {sorted(listed)}, expected {sorted(source.files.values())}"
        )
    extra = sorted(
        p.name
        for p in source.dest.iterdir()
        if p.is_file() and p.name not in listed | {"MANIFEST.json", "README.md"}
    )
    if extra:
        problems.append(f"files not in the manifest: {extra}")
    if problems:
        raise VendorError(f"{source.name}: " + "; ".join(problems))
    return sorted(listed)


def verify_all() -> dict[str, list[str]]:
    return {source.name: verify(source) for source in SOURCES}


def git_output(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(checkout), *args], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def copy_from(source: VendoredSource, checkout: Path) -> None:
    """Copy the pinned files from a source checkout and rewrite the manifest."""
    head = git_output(checkout, "rev-parse", "HEAD")
    if head != source.pinned_commit:
        raise VendorError(f"{checkout} is at {head}, expected {source.pinned_commit}")
    dirty = git_output(checkout, "status", "--porcelain", "--", *source.files)
    if dirty:
        raise VendorError(f"{checkout} has uncommitted changes to vendored files:\n{dirty}")
    source.dest.mkdir(parents=True, exist_ok=True)
    entries = []
    for src_rel, dest_name in source.files.items():
        shutil.copyfile(checkout / src_rel, source.dest / dest_name)
        entries.append(
            {
                "path": dest_name,
                "source_path": src_rel,
                "sha256": sha256(source.dest / dest_name),
            }
        )
    manifest = {
        "dataset": source.description,
        "license": source.license,
        "source_repo": source.source_repo,
        "source_commit": source.pinned_commit,
        "files": entries,
    }
    (source.dest / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
