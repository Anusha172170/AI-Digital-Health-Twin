"""Versioned clinical model experiments; no changes to baseline inference artifacts."""
import hashlib
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_recall_curve
from xgboost import XGBClassifier
from ml_engine.src.config import BASE, PROCESSED, MODELS, TASKS, SEED
from ml_engine.src.evaluate_models import evaluate

OUT = BASE / 'experiments' / 'v2'


def candidates():
    return {
        'logistic': (LogisticRegression(max_iter=3000, random_state=SEED),
            {'model__C': [.01,.1,1,10,100], 'model__class_weight': [None,'balanced']}),
        'random_forest': (RandomForestClassifier(n_estimators=250, n_jobs=1, random_state=SEED),
            {'model__max_depth': [3,6,None], 'model__min_samples_leaf': [3,10,25], 'model__class_weight': [None,'balanced']}),
        'extra_trees': (ExtraTreesClassifier(n_estimators=250, n_jobs=1, random_state=SEED),
            {'model__max_depth': [4,8,None], 'model__min_samples_leaf': [3,10,25], 'model__class_weight': [None,'balanced']}),
        'hist_gradient': (HistGradientBoostingClassifier(random_state=SEED, max_iter=150, early_stopping=False),
            {'model__learning_rate': [.03,.07], 'model__max_leaf_nodes': [7,15], 'model__l2_regularization': [1,10], 'model__min_samples_leaf': [15,30]}),
        'xgboost': (XGBClassifier(n_estimators=200, n_jobs=1, random_state=SEED, tree_method='hist', eval_metric='logloss'),
            {'model__max_depth': [2,3,4], 'model__learning_rate': [.02,.05], 'model__reg_lambda': [5,20],
             'model__min_child_weight': [3,10], 'model__subsample': [.8,1], 'model__scale_pos_weight': [1,3]})}


def threshold_for(y, p):
    precision, recall, thresholds = precision_recall_curve(y, p)
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[np.argmax(scores)])


def run(task):
    spec = TASKS[task]
    dest = OUT / task
    dest.mkdir(parents=True, exist_ok=True)
    # Read train/validation only until selection and calibration choices are frozen.
    tr = pd.read_csv(PROCESSED / task / 'train.csv')
    va = pd.read_csv(PROCESSED / task / 'validation.csv')
    X, y = tr[spec['features']], tr[spec['target']]
    V, vy = va[spec['features']], va[spec['target']]
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    old = joblib.load(MODELS / f'{task}_model.pkl')
    baseline = Pipeline([('preprocessor', old['preprocessor']), ('model', old['model'])])
    fitted = {'baseline': baseline}
    validation = {'baseline': evaluate(vy, baseline.predict_proba(V)[:,1], old['threshold'])}
    searches = {}
    for name, (estimator, params) in candidates().items():
        pipe = Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True)),
                         ('scaler', StandardScaler()), ('model', estimator)])
        search = RandomizedSearchCV(pipe, params, n_iter=8, scoring='average_precision',
            cv=cv, n_jobs=2, random_state=SEED, refit=True, error_score='raise')
        search.fit(X, y)
        pd.DataFrame(search.cv_results_).to_csv(dest / f'{name}_cv.csv', index=False)
        fitted[name] = search.best_estimator_
        p = fitted[name].predict_proba(V)[:,1]
        validation[name] = evaluate(vy, p, threshold_for(vy,p))
        searches[name] = {'parameters': search.best_params_, 'cv_average_precision': search.best_score_}
        joblib.dump({'pipeline': fitted[name], 'features': spec['features'],
                     'task': task, 'threshold': validation[name]['threshold'],
                     'meaning': spec['meaning'], 'search': searches[name],
                     'warning': 'Uncalibrated research candidate; not the selected deployment model'},
                    dest / f'{name}_candidate.pkl', compress=3)
        print(task, name, 'validation AP:', validation[name]['average_precision'], flush=True)
    selected = max(validation, key=lambda n: validation[n]['average_precision'])
    raw = fitted[selected]
    # Refit chosen configuration inside calibration folds on TRAIN ONLY.
    calibrated = CalibratedClassifierCV(raw, method='sigmoid', cv=cv, ensemble=False, n_jobs=2)
    calibrated.fit(X,y)
    cp = calibrated.predict_proba(V)[:,1]
    cm = evaluate(vy,cp,threshold_for(vy,cp))
    rm = validation[selected]
    use_cal = cm['brier_score'] < rm['brier_score'] and cm['average_precision'] >= rm['average_precision'] - .01
    chosen = calibrated if use_cal else raw
    vp = chosen.predict_proba(V)[:,1]
    threshold = threshold_for(vy,vp)
    decision = {'selected_family': selected, 'sigmoid_calibration_selected': bool(use_cal),
        'threshold': threshold, 'validation': validation, 'calibrated_validation': cm, 'search': searches,
        'selection_policy': 'Maximum validation AP including baseline. Accept train-CV sigmoid calibration only if validation Brier improves and AP loss <= 0.01. Max-validation-F1 threshold.',
        'test_caveat': 'Previously inspected historical test split; exploratory comparison, not fresh external validation.'}
    (dest / 'selection.json').write_text(json.dumps(decision,indent=2)+'\n')
    bundle = {'pipeline': chosen, 'features': spec['features'], 'task': task,
              'threshold': threshold, 'meaning': spec['meaning'], 'selection': decision,
              'warning': 'Research only; no clinical/external validation. Calibration is not certification.'}
    joblib.dump(bundle,dest / 'model_bundle.pkl',compress=3)
    # Selection is complete. Read historical test set only now.
    test = pd.read_csv(PROCESSED / task / 'test.csv')
    T, ty = test[spec['features']], test[spec['target']]
    p = chosen.predict_proba(T)[:,1]
    bp = baseline.predict_proba(T)[:,1]
    metrics = {'baseline': evaluate(ty,bp,old['threshold']), 'candidate_v2': evaluate(ty,p,threshold)}
    pd.DataFrame({'source_row_id':test.source_row_id,'actual':ty,'baseline_probability':bp,
        'candidate_probability':p,'candidate_prediction':(p>=threshold).astype(int)}).to_csv(dest/'test_predictions.csv',index=False)
    # Paired bootstrap uncertainty: resample rows, not independent cohorts.
    from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
    rng = np.random.default_rng(SEED)
    deltas = {'roc_auc': [], 'average_precision': [], 'brier_score': []}
    yy = ty.to_numpy()
    for _ in range(1000):
        idx = rng.integers(0,len(yy),len(yy))
        if len(np.unique(yy[idx])) < 2:
            continue
        for name, metric in [('roc_auc',roc_auc_score),('average_precision',average_precision_score),('brier_score',brier_score_loss)]:
            deltas[name].append(metric(yy[idx],p[idx])-metric(yy[idx],bp[idx]))
    metrics['paired_bootstrap_delta_95ci'] = {k: np.quantile(v,[.025,.975]).tolist() for k,v in deltas.items()}
    from sklearn.metrics import classification_report
    metrics['candidate_classification_report'] = classification_report(
        ty, p >= threshold, labels=[0, 1], target_names=['negative', 'positive'],
        output_dict=True, zero_division=0)
    (dest/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    pd.DataFrame({'actual':vy,'probability':vp}).to_csv(dest/'validation_predictions.csv',index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.calibration import CalibrationDisplay
    from sklearn.metrics import PrecisionRecallDisplay
    fig, axes = plt.subplots(1,2,figsize=(10,4))
    for label, pp in [('baseline',bp),('v2',p)]:
        PrecisionRecallDisplay.from_predictions(ty,pp,name=label,ax=axes[0])
        CalibrationDisplay.from_predictions(ty,pp,name=label,n_bins=8,ax=axes[1])
    fig.suptitle(f'{task}: historical test comparison')
    fig.tight_layout()
    fig.savefig(dest/'comparison.png',dpi=130)
    plt.close(fig)
    # Model-agnostic permutation importance explains selected pipeline, not old SHAP.
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(chosen,V,vy,scoring='average_precision',n_repeats=5,random_state=SEED,n_jobs=2)
    pd.DataFrame({'feature':spec['features'],'validation_AP_drop_mean':imp.importances_mean,
        'std':imp.importances_std}).sort_values('validation_AP_drop_mean',ascending=False).to_csv(dest/'permutation_importance.csv',index=False)
    print(task,metrics,flush=True)
    return decision,metrics


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    results = {task: run(task) for task in TASKS}
    lines = ['# V2 clinical model experiments','',
        'Baseline models are unchanged. Selection is based on training CV and validation only. The historical test set has been seen before; all comparisons remain exploratory.', '',
        '| Task | V2 selection | ROC-AUC old → v2 | AP old → v2 | Precision old → v2 | Recall old → v2 | Brier old → v2 |',
        '|---|---|---|---|---|---|---|']
    for task,(d,m) in results.items():
        pairs = [' → '.join(f'{m[n][k]:.3f}' for n in ['baseline','candidate_v2']) for k in ['roc_auc','average_precision','precision','recall','brier_score']]
        lines.append('| '+task+' | '+d['selected_family']+(' + sigmoid' if d['sigmoid_calibration_selected'] else '')+' | '+' | '.join(pairs)+' |')
    lines += ['', 'AP = average precision; higher is better. Brier = probability squared error; lower is better. Threshold metrics are not directly comparable without their precision/recall tradeoff.',
        'Each metrics.json includes 1,000 paired row-bootstrap 95% intervals for v2-minus-baseline changes. These do not include training/model-selection uncertainty or prove clinical benefit.',
        'Preprocessing is refitted inside each training CV fold. No SMOTE before splitting, invented clinical features, new label proxies or test-driven candidate selection.',
        'The selected pipeline is explained using validation permutation importance. Baseline XGBoost SHAP files do NOT explain these new ensembles/calibrated models.',
        'Prospective hypertension, biological age and causal what-if outcomes remain unsupported. Wearable activity models are unchanged.']
    (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
    hashes = {str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.glob('*/*.pkl')}
    (OUT/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')

if __name__ == '__main__':
    main()
