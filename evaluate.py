"""Outcome-proxy evaluation and strict original-name submission contracts."""
import csv
from datetime import date, datetime
import math
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from routing.data import TEAMS


def split_by_time(records: list[dict]) -> dict:
    result = {'train': [], 'dev': [], 'holdout': []}
    for row in records:
        timestamp = row['created_at_ist']
        try:
            created_date = datetime.fromisoformat(timestamp).date()
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid created_at_ist timestamp: {timestamp!r}') from exc
        if date(2025, 4, 1) <= created_date < date(2026, 1, 1):
            result['train'].append(row)
        elif date(2026, 1, 1) <= created_date < date(2026, 4, 1):
            result['dev'].append(row)
        elif date(2026, 4, 1) <= created_date < date(2026, 7, 1):
            result['holdout'].append(row)
        else:
            raise ValueError(f'Unexpected training date: {created_date}')
    return result


def wilson(correct, count):
    if not count:
        return None
    p = correct/count; z = 1.959963984540054
    center = (p+z*z/(2*count))/(1+z*z/count)
    width = z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/(1+z*z/count)
    return [center-width, center+width]


def evaluate_predictions(truth: list[str], predicted: list[str]) -> dict:
    if not truth or len(truth) != len(predicted):
        raise ValueError('Nonempty aligned predictions required')
    if set(truth + predicted) - set(TEAMS):
        raise ValueError('Unknown canonical label')
    precision, recall, f1, support = precision_recall_fscore_support(truth, predicted, labels=list(TEAMS), zero_division=0)
    correct = sum(a == b for a,b in zip(truth,predicted))
    return {'n': len(truth), 'accuracy': float(accuracy_score(truth,predicted)), 'macro_f1': float(f1_score(truth,predicted,labels=list(TEAMS),average='macro',zero_division=0)), 'wilson_95': wilson(correct,len(truth)), 'labels': list(TEAMS), 'confusion_matrix': confusion_matrix(truth,predicted,labels=list(TEAMS)).tolist(), 'per_team': {team: {'precision': float(precision[i]),'recall': float(recall[i]),'f1': float(f1[i]),'support': int(support[i])} for i,team in enumerate(TEAMS)}}


def write_predictions(ids: list[str], teams: list[str], path: Path) -> None:
    if len(ids) != len(teams) or len(ids) != len(set(ids)) or any(not x.strip() for x in ids) or set(teams)-set(TEAMS):
        raise ValueError('Aligned unique nonempty IDs and original team names required')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['request_id','team'])
        writer.writerows(zip(ids,teams))
