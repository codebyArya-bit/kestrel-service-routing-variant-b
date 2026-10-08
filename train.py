"""Reproduce private temporal selection, one frozen evaluation and submission.

Run: OPENBLAS_NUM_THREADS=1 python train.py --data private-data --output private-output
Never publish data, artifacts, metrics, manifests or row-level errors.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from evaluate import evaluate_predictions, split_by_time, wilson, write_predictions
from routing.data import canonical_team, clean_text, input_signature, load_pack
from routing.model import fit_model, save_model, nonactionable


def policy_route(record):
    """Declared policy-aware rules; payment mention alone never triggers Billing."""
    text = clean_text(record['request_text'])
    rules = [('Billing',r'invoice|gst|double charg|charged twice|refund|emi|payment (?:fail|error|issue)|debited'), ('Returns & Replacement',r'return|replacement|replace|exchange|arrived damaged|wrong model|used .*want|delivery|delivered'), ('Repairs',r'not working|stopped working|not turning|noise|leak|burn|blank|fault|repair|broken|error'), ('Warranty Claims',r'warranty|shield|renew|registration|register|claim'), ('Installations',r'install|demo|setup'), ('Consumables',r'filter|cartridge|consumable')]
    return next((team for team,pattern in rules if re.search(pattern,text)), 'Product Advice')


def scored(model, records):
    probs = model.classifier.predict_proba(model.features(records))
    order = np.argsort(-probs,axis=1)
    pred = [str(model.classifier.classes_[i]) for i in order[:,0]]
    rows = np.arange(len(records))
    return pred, probs[rows,order[:,0]], probs[rows,order[:,0]]-probs[rows,order[:,1]]


def review_tradeoff(records, truth, pred, scores, margins):
    actionable = np.array([not nonactionable(r) for r in records])
    correct = np.array([a==b for a,b in zip(truth,pred)])
    rows=[]
    for score in [0,.2,.3,.4,.5,.6,.7,.8,.85,.9,.95]:
        for margin in [0,.05,.1,.15,.2,.3,.4,.5,.6,.7,.8]:
            accept = actionable & (scores>=score) & (margins>=margin)
            n=int(accept.sum()); c=int(correct[accept].sum())
            rows.append({'score_threshold':score,'margin_threshold':margin,'accepted':n,'coverage':n/len(records),'accuracy':c/n if n else None,'wilson_95':wilson(c,n)})
    eligible=[r for r in rows if r['accepted'] and r['accuracy']>=.90]
    selected=max(eligible,key=lambda r:(r['coverage'],r['accuracy'],-r['score_threshold'],-r['margin_threshold'])) if eligible else max(rows,key=lambda r:(r['accuracy'] or 0,r['coverage']))
    return {'criterion':'accepted outcome accuracy >= 0.90; maximize development coverage','criterion_met':bool(eligible),'selected':selected,'grid':rows,'nonactionable_review_count':int((~actionable).sum())}


def details(records,truth,pred,fit_records):
    result=evaluate_predictions(truth,pred)
    signatures={input_signature(r) for r in fit_records}
    groups={}
    for field in ['month','channel','product_family','signature']:
        values=[r['created_at_ist'][:7] if field=='month' else ('seen' if input_signature(r) in signatures else 'unseen') if field=='signature' else r[field] for r in records]
        groups[field]={}
        for value in sorted(set(values)):
            indices=[i for i,v in enumerate(values) if v==value]
            groups[field][value]=evaluate_predictions([truth[i] for i in indices],[pred[i] for i in indices])
    result['segments']=groups
    result['bot_agreement']=sum(p==canonical_team(r['team_label']) for p,r in zip(pred,records))/len(records)
    return result


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('private-data'))
    parser.add_argument('--output',type=Path,default=Path('private-output'))
    args=parser.parse_args(); start=time.monotonic()
    args.output.mkdir(parents=True,exist_ok=True)
    # Guard accidental replacement of an existing frozen experiment.
    if (args.output/'freeze.json').exists():
        raise RuntimeError('Existing frozen experiment: select a new private output directory')
    pack=load_pack(args.data)
    outcomes={r['request_id']:canonical_team(r['final_team']) for r in pack['resolution']}
    split=split_by_time(pack['train']); train,dev=split['train'],split['dev']
    labels=lambda records:[outcomes[r['request_id']] for r in records]
    ytrain,ydev=labels(train),labels(dev)
    development={}; candidates=[]
    for use_char,use_metadata in [(False,False),(False,True),(True,False),(True,True)]:
        name=f'word{"char" if use_char else ""}{"_metadata" if use_metadata else ""}'
        cfg={'use_char':use_char,'use_metadata':use_metadata,'C':4.,'max_iter':1000,'random_state':42,'mode':'private'}
        model=fit_model(train,ytrain,cfg)
        pred,scores,margins=scored(model,dev)
        metrics=evaluate_predictions(ydev,pred)
        development[name]=metrics
        candidates.append((name,cfg,model,pred,scores,margins,metrics))
        print('development',name,metrics['accuracy'],metrics['macro_f1'],flush=True)
    selected=max(candidates,key=lambda c:(c[6]['accuracy'],c[6]['macro_f1'],-int(c[1]['use_char'])-int(c[1]['use_metadata']),-int(c[1]['use_char'])))
    name,cfg,_,devpred,scores,margins,_=selected
    threshold=review_tradeoff(dev,ydev,devpred,scores,margins)
    cfg.update({k:threshold['selected'][k] for k in ['score_threshold','margin_threshold']})
    majority=Counter(ytrain).most_common(1)[0][0]
    development['majority']=evaluate_predictions(ydev,[majority]*len(dev))
    development['policy']=evaluate_predictions(ydev,[policy_route(r) for r in dev])
    development['current_bot']=evaluate_predictions(ydev,[canonical_team(r['team_label']) for r in dev])
    botmodel=fit_model(train,[canonical_team(r['team_label']) for r in train],cfg)
    development['bot_trained']=evaluate_predictions(ydev,scored(botmodel,dev)[0])
    # This immutable manifest is written BEFORE any holdout predictions or metrics.
    freeze={'selected_model':name,'config':cfg,'selection':'development all-row accuracy, macro-F1 tie-break, fewer feature blocks then word-only','threshold_selection':threshold,'development':development,'fit_count':len(train)+len(dev),'fit_latest_date':max(r['created_at_ist'] for r in train+dev),'holdout_boundary':'2026-04-01 to 2026-06-30 inclusive','input_sha256':{p.name:sha(p) for p in sorted(args.data.glob('*.csv'))},'source_sha256':{p:sha(p) for p in ['train.py','evaluate.py','routing/model.py','routing/data.py']},'random_state':42}
    freeze_path=args.output/'freeze.json';freeze_path.write_text(json.dumps(freeze,indent=2)+'\n')
    freeze_hash=sha(freeze_path);(args.output/'freeze.sha256').write_text(freeze_hash+'\n')
    print('FROZEN',name,freeze_hash,threshold['selected'],flush=True)
    fit_records=train+dev; model=fit_model(fit_records,labels(fit_records),cfg)
    holdout=split['holdout'];yhold=labels(holdout)
    pred,hscores,hmargins=scored(model,holdout)
    metrics=details(holdout,yhold,pred,fit_records)
    accepted=np.array([not nonactionable(r) for r in holdout]) & (hscores>=cfg['score_threshold']) & (hmargins>=cfg['margin_threshold'])
    metrics['accepted']={'count':int(accepted.sum()),'coverage':float(accepted.mean()),'accuracy':float(np.mean([yhold[i]==pred[i] for i in np.where(accepted)[0]])) if accepted.any() else None}
    metrics['baselines']={'majority':evaluate_predictions(yhold,[majority]*len(holdout)),'policy':evaluate_predictions(yhold,[policy_route(r) for r in holdout]),'current_bot':evaluate_predictions(yhold,[canonical_team(r['team_label']) for r in holdout])}
    bot_final=fit_model(fit_records,[canonical_team(r['team_label']) for r in fit_records],cfg)
    metrics['baselines']['bot_trained']=evaluate_predictions(yhold,scored(bot_final,holdout)[0])
    records=pack['train'];groups=[input_signature(r) for r in records]
    group_train,group_test=next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(records,groups=groups))
    sensitivity_train=[records[i] for i in group_train];sensitivity_test=[records[i] for i in group_test]
    sensitivity_model=fit_model(sensitivity_train,labels(sensitivity_train),cfg)
    metrics['grouped_signature_sensitivity']=evaluate_predictions(labels(sensitivity_test),scored(sensitivity_model,sensitivity_test)[0])
    metrics['grouped_signature_sensitivity'].update({'train_count':len(sensitivity_train),'test_count':len(sensitivity_test),'signature_overlap':len({groups[i] for i in group_train}&{groups[i] for i in group_test}),'seed':42,'purpose':'secondary sensitivity; no temporal deployment interpretation; no tuning'})
    signatures={}
    for r in records:signatures.setdefault(input_signature(r),set()).add(outcomes[r['request_id']])
    metrics['conflicting_signature_groups']=sum(len(v)>1 for v in signatures.values())
    metrics['forecast']={'rounded_accuracy_percent':round(metrics['accuracy']*100),'plausible_range_percent':[max(0,round(metrics['accuracy']*100)-5),min(100,round(metrics['accuracy']*100)+5)],'basis':'frozen April-June outcome accuracy; plausible range is judgmental drift allowance, not confidence interval','limitations':'Final closing team is an operational proxy, not independently adjudicated intent. Ambiguous requests and conflicting signatures retained. Hidden distribution and label semantics may differ.'}
    with (args.output/'holdout_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['request_id','created_at_ist','channel','product_family','request_text','truth','prediction']);writer.writeheader()
        for r,t,p in zip(holdout,yhold,pred):
            if t!=p:writer.writerow({**{k:r[k] for k in writer.fieldnames if k in r},'truth':t,'prediction':p})
    final_model=fit_model(records,labels(records),cfg);save_model(final_model,args.output/'model.pkl')
    final_pred=scored(final_model,pack['test'])[0]
    ids=[r['request_id'] for r in pack['test']]
    if len(ids)!=len(pack['sample_submission']) or ids != [r['request_id'] for r in pack['sample_submission']]:raise ValueError('Submission length/order mismatch')
    submission=args.output/'predictions.csv';write_predictions(ids,final_pred,submission)
    if sha(freeze_path)!=freeze_hash:raise RuntimeError('Frozen manifest altered')
    metrics.update({'freeze_sha256':freeze_hash,'submission_sha256':sha(submission),'model_sha256':sha(args.output/'model.pkl'),'submission_count':len(ids),'final_fit_count':len(records),'runtime_seconds':time.monotonic()-start})
    (args.output/'evaluation.json').write_text(json.dumps(metrics,indent=2)+'\n')
    print('HOLDOUT',metrics['accuracy'],metrics['macro_f1'],'ACCEPTED',metrics['accepted'],'FORECAST',metrics['forecast'],'RUNTIME',metrics['runtime_seconds'],flush=True)

if __name__=='__main__':main()
