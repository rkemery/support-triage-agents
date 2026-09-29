"""Verify (or refresh) the vendored data in data/tallowbrook/ and data/rag_snapshot/.

    uv run python scripts/sync_data.py    # verify every vendored file against its MANIFEST.json
    uv run python scripts/sync_data.py --tallowbrook ../rag-support-assistant  # at tallowbrook-v0.1
    uv run python scripts/sync_data.py --rag ../rag-support-assistant  # at the snapshot commit

Copy mode refuses a checkout that is not at the pinned commit or has uncommitted
changes to the vendored files. The Tallowbrook dataset's canonical home will be
a Hugging Face dataset. Until then the source is a git checkout.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from support_triage_agents.vendor import (
    RAG_SNAPSHOT,
    TALLOWBROOK,
    VendorError,
    copy_from,
    verify_all,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--tallowbrook",
        type=Path,
        help="rag-support-assistant checkout at tag tallowbrook-v0.1",
    )
    parser.add_argument("--rag", type=Path, help="rag-support-assistant checkout")
    args = parser.parse_args(argv)
    try:
        if args.tallowbrook is not None:
            copy_from(TALLOWBROOK, args.tallowbrook)
        if args.rag is not None:
            copy_from(RAG_SNAPSHOT, args.rag)
        verified = verify_all()
    except VendorError as exc:
        print(f"sync_data: {exc}", file=sys.stderr)
        return 1
    for name, files in verified.items():
        print(f"sync_data: {name}: {len(files)} files match MANIFEST.json ({', '.join(files)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
