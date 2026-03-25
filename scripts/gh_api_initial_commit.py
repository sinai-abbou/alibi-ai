#!/usr/bin/env python3
"""Create initial (or follow-up) commit on GitHub via REST API when local git is unavailable."""

from __future__ import annotations

import base64
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import quote

OWNER = "sinai-abbou"
REPO = "alibi-ai"
COMMIT_MESSAGE = "chore: initial project scaffold (backend, frontend, tooling)"

SKIP_DIR_NAMES = frozenset(
    {
        ".git",
        ".venv",
        "venv",
        "__pycache__",
        ".mypy_cache",
        ".ruff_cache",
        ".pytest_cache",
        "dist",
        "build",
        "node_modules",
    }
)
SKIP_SUFFIXES = frozenset({".pyc", ".pyo", ".pyd"})
SKIP_EXACT = frozenset({"backend/data/embeddings_cache.json", ".DS_Store"})


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def gh_token() -> str:
    return subprocess.check_output(["gh", "auth", "token"], text=True).strip()


def api(token: str, method: str, path: str, data: dict | None = None) -> dict:
    url = f"https://api.github.com{path}"
    payload = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=payload, method=method)
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("X-GitHub-Api-Version", "2022-11-28")
    if payload is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as resp:
            body = resp.read().decode()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        detail = e.read().decode()
        raise RuntimeError(f"{method} {path} HTTP {e.code}: {detail}") from e


def get_main_tip_sha(token: str) -> str | None:
    try:
        ref = api(token, "GET", f"/repos/{OWNER}/{REPO}/git/ref/heads/main")
        return str(ref["object"]["sha"])
    except RuntimeError as e:
        err = str(e)
        # Empty GitHub repos return 409 on git refs until the first commit exists.
        if "HTTP 404" in err or "HTTP 409" in err:
            return None
        raise


def seed_empty_repository(token: str) -> None:
    """GitHub returns 409 on git/blobs until the repo has at least one commit."""
    stub = b"# Alibi AI\n"
    path = quote("README.md")
    api(
        token,
        "PUT",
        f"/repos/{OWNER}/{REPO}/contents/{path}",
        {
            "message": "chore: bootstrap empty repository for git API",
            "content": base64.b64encode(stub).decode("ascii"),
        },
    )


def commit_tree_sha(token: str, commit_sha: str) -> str:
    commit = api(token, "GET", f"/repos/{OWNER}/{REPO}/git/commits/{commit_sha}")
    return str(commit["tree"]["sha"])


def list_tracked_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for p in root.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(root)
        rel_s = rel.as_posix()
        if rel_s in SKIP_EXACT or p.name == ".DS_Store":
            continue
        if any(part in SKIP_DIR_NAMES or part.endswith(".egg-info") for part in rel.parts):
            continue
        if p.suffix in SKIP_SUFFIXES:
            continue
        out.append(rel)
    return sorted(out)


def main() -> int:
    root = repo_root()
    token = gh_token()
    files = list_tracked_files(root)
    if not files:
        print("No files to commit.", file=sys.stderr)
        return 1

    parent_sha = get_main_tip_sha(token)
    if parent_sha is None:
        seed_empty_repository(token)
        parent_sha = get_main_tip_sha(token)
    if parent_sha is None:
        print("Could not resolve main after bootstrap.", file=sys.stderr)
        return 1

    base_tree = commit_tree_sha(token, parent_sha)

    tree_items: list[dict[str, str]] = []
    for rel in files:
        raw = (root / rel).read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        blob = api(
            token,
            "POST",
            f"/repos/{OWNER}/{REPO}/git/blobs",
            {"content": b64, "encoding": "base64"},
        )
        tree_items.append(
            {
                "path": rel.as_posix(),
                "mode": "100644",
                "type": "blob",
                "sha": blob["sha"],
            }
        )

    tree = api(
        token,
        "POST",
        f"/repos/{OWNER}/{REPO}/git/trees",
        {"base_tree": base_tree, "tree": tree_items},
    )

    commit = api(
        token,
        "POST",
        f"/repos/{OWNER}/{REPO}/git/commits",
        {"message": COMMIT_MESSAGE, "tree": tree["sha"], "parents": [parent_sha]},
    )

    api(
        token,
        "PATCH",
        f"/repos/{OWNER}/{REPO}/git/refs/heads/main",
        {"sha": commit["sha"]},
    )

    print(f"Updated {OWNER}/{REPO} main -> {commit['sha'][:7]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
