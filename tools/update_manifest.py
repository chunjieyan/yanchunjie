#!/usr/bin/env python3
"""Rebuild the release-byte manifest after intentional, reviewed source edits."""
from pathlib import Path
import argparse
import json
from validate_release import ROOT, digest, release_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Write the new manifest; default prints a change summary only.')
    args = parser.parse_args()
    path = ROOT / 'MANIFEST.sha256.json'
    old = json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}
    entries = {name: {'sha256': digest(file), 'bytes': file.stat().st_size} for name, file in sorted(release_files().items())}
    previous = old.get('files', {})
    changed = [name for name in sorted(entries) if previous.get(name) != entries[name]]
    removed = sorted(previous.keys() - entries.keys())
    manifest = {
        'schema': 'KAIS_CODE_CANDIDATE_FILE_SHA256_V1',
        'date': '2026-11-01',
        'status': 'LOCAL_PREPARED_RELEASE_NOT_UPLOADED',
        'self_excluded': 'MANIFEST.sha256.json',
        'generated_files_excluded': 'outputs/, __pycache__/, .git/, .venv/, .pytest_cache/',
        'files': entries,
    }
    if args.write:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'MANIFEST_WRITTEN' if args.write else 'MANIFEST_CHANGE_PLAN_ONLY', 'file_count': len(entries), 'changed_or_added': changed, 'removed': removed, 'scientific_source_hashes_in_provenance_rewritten': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
