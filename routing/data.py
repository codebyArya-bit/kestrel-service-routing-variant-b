"""Private CSV contracts and inference-safe text normalization (standard library)."""
import csv
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

TEAMS = ('Installations', 'Repairs', 'Consumables', 'Billing', 'Returns & Replacement', 'Warranty Claims', 'Product Advice')
ALIASES = {'Installs & Demo': 'Installations', 'Filters & Consumables': 'Consumables'}


def canonical_team(name: str) -> str:
    value = ALIASES.get(name.strip(), name.strip())
    if value not in TEAMS:
        raise ValueError(f'Unknown team: {name!r}')
    return value


def clean_text(text: str) -> str:
    """Normalize model text without changing originals; strip request/order numbers."""
    value = str(text)
    # Only reverse mojibake if the reversal actually removes corruption markers.
    if any(marker in value for marker in ('Ã', 'Â', 'â€')):
        try:
            repaired = value.encode('cp1252').decode('utf-8')
            if sum(repaired.count(x) for x in ('Ã', 'Â', 'â€')) < sum(value.count(x) for x in ('Ã', 'Â', 'â€')):
                value = repaired
        except (UnicodeEncodeError, UnicodeDecodeError):
            pass
    value = value.casefold()
    value = re.sub(r'\b(?:sr|ord|order|req|request)[\s#:_-]*\d+[\w-]*\b', ' ', value)
    return ' '.join(value.split())


def input_signature(record: dict) -> str:
    """Stable grouping key for text, product and warranty, excluding join IDs."""
    values = [clean_text(record.get(key, '')) for key in ('request_text', 'product_family', 'warranty_status')]
    return hashlib.sha256(json.dumps(values, ensure_ascii=False).encode('utf-8')).hexdigest()


def _resolve(directory: Path, filename: str) -> Path:
    exact = directory / filename
    if exact.is_file():
        return exact
    matches = list(directory.glob('*-' + filename))
    if len(matches) != 1:
        raise ValueError(f'Expected one {filename}; found {len(matches)}')
    return matches[0]


def load_pack(directory: Path) -> dict[str, list[dict]]:
    """Return unmodified CSV records after checking label and one-to-one contracts.

    Keys: train, test, resolution, teams, sample_submission. Outcomes remain in the
    separate resolution table so callers must explicitly opt into joining targets.
    """
    directory = Path(directory)
    names = {'train': 'train.csv', 'test': 'test_unlabelled.csv', 'resolution': 'resolution_log.csv', 'teams': 'teams.csv', 'sample_submission': 'sample_submission.csv'}
    pack = {}
    for key, filename in names.items():
        with _resolve(directory, filename).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            if len(headers) != len(set(headers)):
                raise ValueError(f'{filename}: duplicate headers')
            intake = {'request_id', 'created_at_ist', 'channel', 'product_family', 'warranty_status', 'request_text', 'source'}
            required = {'teams': {'team', 'renamed_to', 'handles'}, 'train': intake | {'team_label'}, 'test': intake, 'resolution': {'request_id', 'first_team', 'final_team', 'transfers', 'resolved_at'}, 'sample_submission': {'request_id', 'team'}}[key]
            if not required <= set(reader.fieldnames or ()):
                raise ValueError(f'{filename}: missing required columns {sorted(required - set(reader.fieldnames or ()))}')
            rows = list(reader)
        if any(None in row or any(value is None or (not value.strip() and not (key == 'teams' and field == 'renamed_to')) for field, value in row.items()) for row in rows):
            raise ValueError(f'{filename}: blank or malformed cells')
        if key != 'teams':
            ids = [r['request_id'] for r in rows]
            if len(ids) != len(set(ids)):
                raise ValueError(f'Duplicate request_id in {filename}')
        for row in rows:
            for field in ('team', 'team_label', 'first_team', 'final_team'):
                if field in row:
                    canonical_team(row[field])
        pack[key] = rows
    if set(r['team'] for r in pack['teams']) != set(TEAMS) or len(pack['teams']) != len(TEAMS):
        raise ValueError('teams.csv must contain exactly the seven canonical teams')
    train_ids = {r['request_id'] for r in pack['train']}
    test_ids = {r['request_id'] for r in pack['test']}
    if train_ids & test_ids:
        raise ValueError('train/test request_id overlap')
    if train_ids != {r['request_id'] for r in pack['resolution']}:
        raise ValueError('resolution_log coverage must match train one-to-one')
    if [r['request_id'] for r in pack['test']] != [r['request_id'] for r in pack['sample_submission']]:
        raise ValueError('sample_submission template ID order must match test exactly')
    return pack


def audit_pack(pack: dict[str, list[dict]]) -> dict:
    """Operational diagnostics; never alter timestamps or return diagnostic features."""
    resolution = {r['request_id']: r for r in pack['resolution']}
    mismatch = raw_negative = corrected_negative = 0
    signatures = {}
    for row in pack['train']:
        outcome = resolution[row['request_id']]
        final = canonical_team(outcome['final_team'])
        mismatch += canonical_team(row['team_label']) != final
        signatures.setdefault(input_signature(row), set()).add(final)
        created = datetime.fromisoformat(row['created_at_ist'])
        resolved = datetime.fromisoformat(outcome['resolved_at'])
        raw_negative += resolved < created
        corrected = resolved + (timedelta(hours=5, minutes=30) if row['source'] == 'legacy_zoho' else timedelta())
        corrected_negative += corrected < created
    return {'train_rows': len(pack['train']), 'test_rows': len(pack['test']), 'train_test_overlap': len({r['request_id'] for r in pack['train']} & {r['request_id'] for r in pack['test']}), 'bot_outcome_mismatch': mismatch, 'raw_negative_elapsed': raw_negative, 'corrected_negative_elapsed': corrected_negative, 'conflicting_input_groups': sum(len(teams) > 1 for teams in signatures.values())}
