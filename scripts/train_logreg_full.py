#!/usr/bin/env python3
import os, json, zipfile, argparse
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
                             classification_report, confusion_matrix)
import joblib

def load_csv_from_zip(zip_path, csv_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(csv_name) as f:
            return pd.read_csv(f)

def plot_and_save_roc_pr(y_test, y_proba, outdir):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.4f})')
    plt.plot([0,1], [0,1], '--', linewidth=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    roc_path = os.path.join(outdir, 'roc_curve.png')
    plt.tight_layout()
    plt.savefig(roc_path)
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    plt.figure(figsize=(6,4))
    plt.plot(recall, precision, label=f'AP = {ap:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    pr_path = os.path.join(outdir, 'pr_curve.png')
    plt.tight_layout()
    plt.savefig(pr_path)
    plt.close()
    return roc_path, pr_path

def main(args):
    zip_path = args.zip
    csv_name = args.csvname
    outdir = args.outdir
    target_col = args.target
    top_k = args.top_k
    test_size = args.test_size
    random_state = args.random_state
    max_iter = args.max_iter

    Path(outdir).mkdir(parents=True, exist_ok=True)
    print("Loading data...", zip_path, csv_name)
    df = load_csv_from_zip(zip_path, csv_name)
    print("Rows, cols:", df.shape)

    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    if df[target_col].dtype == object:
        df[target_col] = df[target_col].astype(str).str.strip().str.lower().map(
            lambda x: 1 if x in ('1','true','yes','default','charged off','y') else 0)

    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()

    for c in list(cat_cols):
        if df[c].nunique(dropna=True) > top_k:
            top_vals = df[c].value_counts().nlargest(top_k).index
            df[c] = df[c].where(df[c].isin(top_vals), other='__OTHER__')

    num_cols = df.select_dtypes(include=['int64','float64']).columns.tolist()
    if target_col in num_cols:
        num_cols.remove(target_col)
    cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()

    print('Num cols:', len(num_cols), 'Cat cols:', len(cat_cols))
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                        stratify=y, random_state=random_state)
    print('Train/Test:', X_train.shape, X_test.shape)

    num_pipe = Pipeline([('imp', SimpleImputer(strategy='median')), ('sc', StandardScaler())])
    cat_pipe = Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                         ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=True))])
    preproc = ColumnTransformer([('num', num_pipe, num_cols), ('cat', cat_pipe, cat_cols)], sparse_threshold=0)
    clf = Pipeline([('preproc', preproc),
                    ('clf', LogisticRegression(solver='saga', max_iter=max_iter, class_weight='balanced'))])

    print('Fitting model...')
    clf.fit(X_train, y_train)
    print('Fitted.')

    y_proba = clf.predict_proba(X_test)[:,1]
    y_pred = clf.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)
    clf_report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {'roc_auc': float(roc_auc),
               'average_precision': float(ap),
               'confusion_matrix': cm.tolist(),
               'classification_report': clf_report}
    with open(os.path.join(outdir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    roc_path, pr_path = plot_and_save_roc_pr(y_test, y_proba, outdir)

    model_path = os.path.join(outdir, 'logreg_pipeline.joblib')
    joblib.dump(clf, model_path, compress=3)

    try:
        pre = clf.named_steps['preproc']
        ohe = None
        if len(cat_cols) > 0:
            ohe = pre.named_transformers_['cat'].named_steps['ohe']
            cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
        else:
            cat_feature_names = []
        feature_names = list(num_cols) + cat_feature_names
        coefs = clf.named_steps['clf'].coef_[0]
        if len(feature_names) == len(coefs):
            coef_df = pd.DataFrame({'feature': feature_names, 'coef': coefs})
            coef_df['abs_coef'] = coef_df['coef'].abs()
            coef_df = coef_df.sort_values('abs_coef', ascending=False)
            coef_df.to_csv(os.path.join(outdir, 'top_coeffs.csv'), index=False)
    except Exception as e:
        print("Could not extract coefficients:", e)

    print('Done. Results in', outdir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--zip', required=True)
    parser.add_argument('--csvname', default='Loan_Default.csv')
    parser.add_argument('--outdir', default='outputs')
    parser.add_argument('--target', default='Status')
    parser.add_argument('--top_k', type=int, default=50)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--max_iter', type=int, default=2000)
    args = parser.parse_args()
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    main(args)
