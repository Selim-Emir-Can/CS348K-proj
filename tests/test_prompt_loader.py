"""Tests for the hash-locked prompt loader (round 1 §1.1).

The loader's whole point is to abort if a `.txt` was edited without
refreshing the `.json` sha. These tests verify that contract end-to-end.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from prompts.loader import (load_prompt, compute_sha256, rehash,
                              PromptHashMismatch)


def _write_pair(tmp_path: Path, name: str, text: str, sha: str = None) -> Path:
    txt = tmp_path / f'{name}.txt'
    sidecar = tmp_path / f'{name}.json'
    txt.write_text(text, encoding='utf-8')
    payload = {'name': name, 'sha256': sha or compute_sha256(text)}
    sidecar.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return txt


def test_load_prompt_returns_text(tmp_path):
    text = 'system prompt body\nwith multiple lines'
    txt = _write_pair(tmp_path, 'p', text)
    assert load_prompt(txt) == text


def test_load_prompt_raises_on_sha_mismatch(tmp_path):
    """Editing the .txt without rehashing the sidecar is exactly the failure
    mode the lock exists to catch."""
    text = 'original prompt'
    txt = _write_pair(tmp_path, 'p', text)
    txt.write_text('TAMPERED prompt', encoding='utf-8')
    with pytest.raises(PromptHashMismatch):
        load_prompt(txt)


def test_load_prompt_raises_on_missing_sidecar(tmp_path):
    txt = tmp_path / 'p.txt'
    txt.write_text('prompt only; no sidecar', encoding='utf-8')
    with pytest.raises(FileNotFoundError):
        load_prompt(txt)


def test_load_prompt_resolves_relative_to_prompts_dir(tmp_path):
    """The loader resolves a bare filename against prompts/ if it's not
    found at cwd. The shipped round-1 prompt files are exercised here."""
    for name in ('character_llm_prompt.txt',
                 'designer_llm_prompt_open.txt',
                 'designer_llm_prompt_closed.txt',
                 'guided_player_system_prompt.txt'):
        text = load_prompt(name)
        assert isinstance(text, str) and len(text) > 0


def test_rehash_refreshes_sidecar(tmp_path):
    """The escape hatch: editing the prompt deliberately + running rehash
    should leave load_prompt happy again."""
    txt = _write_pair(tmp_path, 'p', 'v1')
    txt.write_text('v2 — deliberate edit', encoding='utf-8')
    meta = rehash(txt)
    assert meta['sha256'] == compute_sha256('v2 — deliberate edit')
    assert load_prompt(txt) == 'v2 — deliberate edit'
