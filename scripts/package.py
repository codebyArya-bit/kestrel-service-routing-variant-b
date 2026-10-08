"""Package explicit public/restricted allowlists; never refit frozen artifacts."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUBLIC_FILES = ('README.md', 'requirements.txt', 'train.py', 'evaluate.py', 'serve.py',
                'demo/synthetic.csv', 'web/index.html', 'scripts/package.py')
PUBLIC_DOCS = ('costs.md', 'decisions.md', 'memo-to-ritu.md', 'prompt-version-log.md',
               'recording-script.md', 'submission-form.md', 'validation.md')
HANDOFF_FILES = ('memo-to-ritu.pdf', 'memo-to-ritu.md', 'submission-form.md', 'status-check.md',
                 'recording-script.md', 'decisions.md', 'validation.md', 'costs.md', 'prompt-version-log.md', 'HANDOFF.md')
FINAL_FILES = ('model.pkl', 'evaluation.json', 'freeze.json', 'freeze.sha256', 'predictions.csv')


def public_paths(root):
    return sorted([root / p for p in PUBLIC_FILES] +
                  [root / 'docs' / p for p in PUBLIC_DOCS] +
                  list((root / 'routing').glob('*.py')) + list((root / 'tests').glob('test_*.py')) +
                  [root / 'routing/__init__.py', root / 'tests/__init__.py'])


def private_markers(root):
    markers = set()
    for name in ('train.csv', 'test_unlabelled.csv', 'resolution_log.csv'):
        with (root / 'private-data' / name).open(newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                markers.add(row['request_id'].encode())
                message = row.get('request_text', '')
                if len(message) >= 80:
                    markers.add(message[:80].encode())
    email = (root / 'private-data/email-thread.txt').read_text()
    policy = subprocess.run(['pdftotext', str(root / 'private-data/ops-policy.pdf'), '-'],
                            check=True, capture_output=True, text=True).stdout
    for document in (email, policy):
        markers.update(line.strip()[:100].encode() for line in document.splitlines() if len(line.strip()) >= 100)
    return markers


def scan_public(paths, markers):
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Unsafe or absent public path: {path.name}')
        content = path.read_bytes()
        if any(marker in content for marker in markers):
            raise ValueError(f'Private content detected in public file: {path.name}')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_frozen_artifacts(root):
    final = root / 'private-output/final'
    evaluation = json.loads((final / 'evaluation.json').read_text())
    model_hash = sha(final / 'model.pkl')
    freeze_hash = sha(final / 'freeze.json')
    if model_hash != evaluation['model_sha256']:
        raise ValueError('Frozen model hash mismatch')
    if freeze_hash != (final / 'freeze.sha256').read_text().strip() or freeze_hash != evaluation['freeze_sha256']:
        raise ValueError('Frozen manifest hash mismatch')
    return {'model_sha256': model_hash, 'freeze_sha256': freeze_hash,
            'evaluation_consistency': True, 'freeze_sidecar_consistency': True}


def check_predictions(root, expected_hash):
    path = root / 'private-output/final/predictions.csv'
    if sha(path) != expected_hash:
        raise ValueError('Frozen prediction hash mismatch')
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['request_id', 'team']:
            raise ValueError('Prediction schema mismatch')
        rows = list(reader)
    def ids(name):
        with (root / 'private-data' / name).open(newline='') as f:
            return [row['request_id'] for row in csv.DictReader(f)]
    actual = [r['request_id'] for r in rows]
    if actual != ids('sample_submission.csv') or actual != ids('test_unlabelled.csv'):
        raise ValueError('Prediction count/order does not match authoritative sample and input')
    teams = {'Installations', 'Repairs', 'Consumables', 'Billing', 'Returns & Replacement', 'Warranty Claims', 'Product Advice'}
    if len(set(actual)) != len(actual) or any(not r['request_id'] or r['team'] not in teams for r in rows):
        raise ValueError('Duplicate/blank IDs or noncanonical team')
    return {'rows': len(rows), 'sha256': expected_hash, 'schema': ['request_id', 'team'],
            'sample_and_input_order': True, 'unique_nonblank_ids': True, 'canonical_teams': True}


def bundle(path, members):
    manifest = {name: sha(source) for name, source in sorted(members.items())}
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, source in sorted(members.items()):
            info = zipfile.ZipInfo(name, (2026, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, source.read_bytes())
        info = zipfile.ZipInfo('manifest-sha256.json', (2026, 1, 1, 0, 0, 0))
        archive.writestr(info, json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    return {'sha256': sha(path), 'file_count': len(members) + 1, 'members': manifest}


def package(root, output, expected_hash):
    paths = sorted(set(public_paths(root)))
    markers = private_markers(root)
    scan_public(paths, markers)
    frozen_checks = check_frozen_artifacts(root)
    checks = check_predictions(root, expected_hash)
    output.mkdir(parents=True, exist_ok=True)
    public = {str(p.relative_to(root)): p for p in paths}
    restricted = dict(public)
    restricted.update({f'final/{name}': root / 'private-output/final' / name for name in FINAL_FILES})
    restricted.update({f'handoff/{name}': root / 'private-output/handoff' / name for name in HANDOFF_FILES})
    reports = {'public-safe.zip': bundle(output / 'public-safe.zip', public),
               'restricted-working.zip': bundle(output / 'restricted-working.zip', restricted)}
    for name, source in {'predictions.csv': root / 'private-output/final/predictions.csv',
                         'memo-to-ritu.pdf': root / 'private-output/handoff/memo-to-ritu.pdf',
                         'submission-form.md': root / 'private-output/handoff/submission-form.md',
                         'status-check.md': root / 'private-output/handoff/status-check.md',
                         'recording-script.md': root / 'private-output/handoff/recording-script.md'}.items():
        shutil.copyfile(source, output / name)
    reports['frozen_artifact_checks'] = frozen_checks
    reports['csv_checks'] = checks
    reports['public_privacy_checks'] = {'passed': True, 'marker_count': len(markers),
        'method': 'Explicit allowlist plus every real request ID, long intake text prefixes, long supplied email/policy lines',
        'limitations': 'Exact-byte check; does not establish absence of paraphrased private content. Generic docs require human review.'}
    (output / 'package-manifest.json').write_text(json.dumps(reports, indent=2, sort_keys=True) + '\n')
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-prediction-sha256', required=True)
    args = parser.parse_args()
    report = package(ROOT, args.output, args.expected_prediction_sha256)
    print(json.dumps({k: v for k, v in report.items() if k in ('csv_checks', 'public_privacy_checks')}, indent=2))
