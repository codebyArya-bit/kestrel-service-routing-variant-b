"""Inference-safe linear routing. Pickle artifacts must come from trusted local training."""
import pickle
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from routing.data import TEAMS, canonical_team, clean_text

MODEL_VERSION = 'kestrel-linear-v1'
REMIT = {
    'Installations': 'Installation, setup and demonstration appointments.',
    'Repairs': 'Diagnosis and repair of malfunctioning products.',
    'Consumables': 'Filters, cartridges and other consumable supplies.',
    'Billing': 'Invoices, payment errors and payment refunds.',
    'Returns & Replacement': 'Product returns, delivery issues and exchanges.',
    'Warranty Claims': 'Warranty eligibility, registration and claims.',
    'Product Advice': 'Model selection, specifications and purchase advice.',
}
DEFAULT_CONFIG = {'use_char': True, 'use_metadata': False, 'C': 4.0,
                  'max_iter': 1000, 'random_state': 42,
                  'score_threshold': 0.35, 'margin_threshold': 0.10, 'mode': 'demo'}


def nonactionable(record):
    """Conservative text-only guard shared by development and live routing."""
    text = clean_text(record.get('request_text', ''))
    return not re.search(r'install|demo|setup|filter|cartridge|consumable|invoice|gst|charg|refund|emi|payment|repair|not working|stopped|not turning|noise|leak|burn|blank|fault|error|broken|return|replacement|replace|exchange|damaged|deliver|warranty|shield|renew|register|registration|claim|buy|purchase|model|spec|compare|recommend|which|how to|clean|capacity|compatible', text)


def _texts(records):
    return [clean_text(r.get('request_text', '')) for r in records]


def _metadata(records):
    return [' '.join(f'{key}_{clean_text(r.get(key,"unknown")).replace(" ","_")}'
                     for key in ('channel', 'product_family', 'warranty_status')) for r in records]


@dataclass
class RoutingModel:
    config: dict
    word: object
    char: object
    metadata: object
    classifier: object

    def features(self, records):
        blocks = [self.word.transform(_texts(records))]
        if self.char is not None:
            blocks.append(self.char.transform(_texts(records)))
        if self.metadata is not None:
            blocks.append(self.metadata.transform(_metadata(records)))
        return hstack(blocks, format='csr')


def fit_model(records: list[dict], targets: list[str], config: dict) -> object:
    if len(records) != len(targets) or not records:
        raise ValueError('Nonempty records and aligned targets required')
    cfg = dict(DEFAULT_CONFIG, **config)
    labels = [canonical_team(t) for t in targets]
    if len(set(labels)) < 2:
        raise ValueError('At least two target teams required')
    word = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    word.fit(_texts(records))
    char = None
    if cfg['use_char']:
        char = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True)
        char.fit(_texts(records))
    metadata = None
    if cfg['use_metadata']:
        metadata = TfidfVectorizer(token_pattern=r'(?u)\b\w+\b')
        metadata.fit(_metadata(records))
    clf = LogisticRegression(C=cfg['C'], max_iter=cfg['max_iter'], random_state=cfg['random_state'])
    model = RoutingModel(cfg, word, char, metadata, clf)
    clf.fit(model.features(records), labels)
    return model


def route(model: object, record: dict) -> dict:
    x = model.features([record])
    scores = model.classifier.predict_proba(x)[0]
    order = np.argsort(-scores)
    top = int(order[0]); team = str(model.classifier.classes_[top])
    review = bool(nonactionable(record) or scores[top] < model.config['score_threshold'] or scores[top] - scores[order[1]] < model.config['margin_threshold'])
    # Explain evidence favouring the selected class over the closest alternative.
    # This removes common boilerplate and avoids tiny numerical weights as claims.
    coefficients = model.classifier.coef_
    if len(scores) == 2:
        coefficients = np.vstack([-coefficients[0] / 2, coefficients[0] / 2])
    contrast = coefficients[top] - coefficients[int(order[1])]
    contributions = x.toarray()[0] * contrast
    cutoff = max(0.01, float(np.max(np.abs(contributions))) * 0.05)
    reasons = []
    offset = 0
    for kind, vectorizer in [('wording', model.word), ('character fragments', model.char), ('metadata', model.metadata)]:
        if vectorizer is None:
            continue
        names = vectorizer.get_feature_names_out()
        values = contributions[offset:offset + len(names)]
        selected = [int(i) for i in np.argsort(-values) if values[i] >= cutoff][:3]
        if selected and kind != 'metadata':
            phrases = ', '.join('"' + str(names[i]).strip() + '"' for i in selected)
            reasons.append(f'Input evidence: {kind} {phrases} favour {team} over {model.classifier.classes_[order[1]]}.')
        elif selected:
            labels = {'channel': 'channel', 'product_family': 'product family', 'warranty_status': 'warranty status'}
            for i in selected:
                for key, label in labels.items():
                    if str(names[i]).startswith(key + '_'):
                        value = clean_text(record.get(key, 'unknown'))
                        reasons.append(f'Input evidence: {label} "{value}" favours {team} over {model.classifier.classes_[order[1]]}.')
                        break
        offset += len(names)
    reasons.append(f'Team remit: {REMIT[team]}')
    if review:
        if nonactionable(record):
            reasons.append('Human review required: the request lacks recognized actionable routing wording.')
        if scores[top] < model.config['score_threshold']:
            reasons.append('Human review required: model support is below the routing threshold.')
        if scores[top] - scores[order[1]] < model.config['margin_threshold']:
            reasons.append('Human review required: competing teams are close.')
    return {'team': team, 'display_team': {'Installations': 'Installs & Demo', 'Consumables': 'Filters & Consumables'}.get(team, team), 'model_version': MODEL_VERSION,
            'mode': model.config['mode'], 'review_required': review,
            'model_score': float(scores[top]),
            'alternatives': [{'team': str(model.classifier.classes_[i]), 'model_score': float(scores[i])} for i in order[1:3]],
            'reasons': reasons}


def save_model(model: object, path: Path) -> None:
    """Save to a caller-controlled local path; never publish private trained artifacts."""
    with Path(path).open('wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: Path) -> object:
    """Load ONLY caller-verified trusted pickle artifacts (pickle executes code)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Routing model missing: {path}; train or load the synthetic demo explicitly')
    with path.open('rb') as f:
        model = pickle.load(f)
    if not isinstance(model, RoutingModel):
        raise ValueError('Artifact is not a Kestrel routing model')
    return model


def demo_model() -> object:
    """Reproducible public startup model, trained solely on labelled synthetic examples."""
    import csv
    path = Path(__file__).resolve().parent.parent / 'demo' / 'synthetic.csv'
    with path.open(encoding='utf-8', newline='') as f:
        rows = list(csv.DictReader(f))
    return fit_model(rows, [r['team'] for r in rows], {'mode': 'demo'})
