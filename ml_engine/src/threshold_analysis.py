"""Research operating points chosen exclusively on validation data; no deployment change."""
import json
import joblib
import numpy as np
import pandas as pd
from ml_engine.src.config import MODELS, PROCESSED, REPORTS, TASKS
from ml_engine.src.evaluate_models import evaluate

POLICIES = {'recall_80': .80, 'recall_90': .90}

def choose_operating_point(y, probabilities, minimum_recall, minimum_precision=.5):
    """Max specificity subject to BOTH constraints. None if infeasible, not a fallback."""
    table = []
    for threshold in np.unique(np.r_[0., probabilities, 1.]):
        table.append(evaluate(y, probabilities, float(threshold)))
    valid = [m for m in table if m['recall'] >= minimum_recall and m['precision'] >= minimum_precision]
    return max(valid, key=lambda m: (m['specificity'], m['precision'], m['threshold'])) if valid else None


def main():
    dest = REPORTS / 'thresholds'
    dest.mkdir(parents=True, exist_ok=True)
    rows = []
    for task, spec in TASKS.items():
        bundle = joblib.load(MODELS / f'{task}_model.pkl')
        va = pd.read_csv(PROCESSED / task / 'validation.csv')
        def probabilities(df):
            X = bundle['preprocessor'].transform(df[spec['features']])
            return bundle['model'].predict_proba(pd.DataFrame(X,columns=spec['features']))[:,1]
        p = probabilities(va)
        y = va[spec['target']]
        sweep = [evaluate(y,p,t) for t in np.arange(.01,1.,.01)]
        pd.DataFrame(sweep).drop(columns='confusion_matrix').to_csv(dest/f'{task}_validation_sweep.csv',index=False)
        decisions = {'existing_baseline': {'threshold':bundle['threshold']}}
        for name, recall in POLICIES.items():
            decisions[name] = choose_operating_point(y,p,recall)
        # Policies frozen before historical test file is read. No changes based on it.
        (dest/f'{task}_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
        test = pd.read_csv(PROCESSED/task/'test.csv')
        tp = probabilities(test)
        for name, decision in decisions.items():
            if decision is None:
                rows.append({'task':task,'policy':name,'status':'infeasible_on_validation'})
                continue
            m = evaluate(test[spec['target']],tp,decision['threshold'])
            tn,fp = m['confusion_matrix'][0]
            fn,tpos = m['confusion_matrix'][1]
            rows.append({'task':task,'policy':name,'status':'research_only',
                **{k:v for k,v in m.items() if k!='confusion_matrix'},
                'test_cases':len(test),'true_positive':tpos,'false_positive':fp,'false_negative':fn,'true_negative':tn,
                'flags_per_1000_test_cases':1000*(fp+tpos)/len(test)})
    pd.DataFrame(rows).to_csv(dest/'operating_points.csv',index=False)
    text = ['# Research threshold analysis','',
        'No alert thresholds deployed. Baseline models and inference remain unchanged.',
        'Experimental constraints: recall >= 80% or 90%, AND precision >= 50%, selected on validation only. These are research examples, not agreed clinical requirements.',
        'Select maximum specificity among feasible validation thresholds; report infeasible instead of silently relaxing constraints.',
        'Historical test performance is descriptive; constraints satisfied on validation need not hold on test or deployment.', '',
        '| Task | Policy | Status | Test precision | Test recall | Test false positives | Test missed positives |',
        '|---|---|---|---:|---:|---:|---:|']
    for r in rows:
        if r['status'].startswith('infeasible'):
            text.append(f"| {r['task']} | {r['policy']} | Infeasible on validation | — | — | — | — |")
        else:
            text.append(f"| {r['task']} | {r['policy']} | Research only | {r['precision']:.3f} | {r['recall']:.3f} | {r['false_positive']} | {r['false_negative']} |")
    text += ['', 'Flags per 1,000 are per historical test CASES, not per day or per patient stream. No alert-frequency or emergency-detection claim can be made from these tabular data.',
        'Repeated real-time evaluation of long-horizon disease scores must not create repeated emergency alerts. Event deduplication, persistence checks, notification caps, symptoms, clinical review and explicit consent require separate design/validation.']
    (dest/'README.md').write_text('\n'.join(text)+'\n')
    print('\n'.join(text))

if __name__ == '__main__':
    main()
