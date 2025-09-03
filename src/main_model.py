# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re, warnings, os
from typing import List, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')

hierarchy = ['segment','family','class','brick']

FILE_MWPD = 'data/MWPD_FULL.csv'
FILE_CORRECT = 'data/correctly_matched_mapped_gpc.csv'
FILE_PRODUCT_MAP = 'data/product_gpc_mapping.csv'
FILE_VALIDATED = 'data/validated_actually_labeled_test_dataset.csv'

TRAIN_FRAC = 0.70
VAL_FRAC = 0.20
TEST_FRAC = 0.10
RANDOM_SEED = 42

EXCLUDE_SOURCES_FOR_BRICK_TRAIN = {'MWPD_FULL'}

def preprocess_keep_symbols(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\+\-/\.]', ' ', text)
    return ' '.join(text.split())

def normalize_colkey(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", col.strip().lower()) if isinstance(col, str) else ""

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {normalize_colkey(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_colkey(cand)
        if key in norm_map:
            return norm_map[key]
        for k, orig in norm_map.items():
            if key and key in k:
                return orig
    return None

def read_csv_flex(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        print(f"[WARN] File not found: {path}")
        return pd.DataFrame()
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    return pd.read_csv(path, engine="python")

def standardize_label(x):
    if pd.isna(x): return np.nan
    s = str(x).lower()
    s = re.sub(r'[0-9]+', ' ', s)
    s = re.sub(r'[^\w\s]', ' ', s)
    s = re.sub(r'[_\-\/&]+', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s if s else np.nan

def standardize_df(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["product_name","description","segment","family","class","brick","source"])
    product_candidates = ["product_name","name","translated_name","item_name","title","item","product"]
    desc_candidates = ["description","desc","details"]
    segment_candidates = ["segment","segmenttitle","segment_title","predicted_segment"]
    family_candidates  = ["family","familytitle","family_title","predicted_family"]
    class_candidates   = ["class","classtitle","class_title","predicted_class","categoryclass"]
    brick_candidates   = ["brick","bricktitle","brick_title","predicted_brick","gpc_brick"]
    pcol = find_col(df, product_candidates)
    dcol = find_col(df, desc_candidates)
    scol = find_col(df, segment_candidates)
    fcol = find_col(df, family_candidates)
    ccol = find_col(df, class_candidates)
    bcol = find_col(df, brick_candidates)
    out = pd.DataFrame()
    if pcol:
        out['product_name'] = df[pcol].astype(str)
    else:
        out['product_name'] = np.nan
    out['description'] = df[dcol].astype(str) if dcol else np.nan
    out['segment'] = df[scol].astype(str) if scol else np.nan
    out['family']  = df[fcol].astype(str) if fcol else np.nan
    out['class']   = df[ccol].astype(str) if ccol else np.nan
    out['brick']   = df[bcol].astype(str) if bcol else np.nan
    for col in ['product_name','description','segment','family','class','brick']:
        out[col] = out[col].astype(str).map(lambda x: re.sub(r"\s+"," ",x).strip() if isinstance(x,str) else x)
        out[col] = out[col].replace("", np.nan).replace("nan", np.nan)
    for col in ['segment','family','class','brick']:
        out[col] = out[col].map(standardize_label)
    out['source'] = source_name
    out = out[~out['product_name'].isna() & (out['product_name'].str.strip()!="")].reset_index(drop=True)
    return out

def make_dedup_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\W_]+", " ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def vec_builder():
    return TfidfVectorizer(
        max_features=15000,
        ngram_range=(1,2),
        min_df=2,
        max_df=0.95,
        stop_words='english',
        sublinear_tf=True,
        norm='l2'
    )

def model_builder():
    return LinearSVC(C=1.0, class_weight='balanced')

def train_single_level(X_train, y_series):
    le = LabelEncoder()
    y = y_series.astype(str).fillna("NA_LBL")
    y_enc = le.fit_transform(y)
    if len(np.unique(y_enc)) < 2:
        model = ('const', int(y_enc[0]))
    else:
        clf = model_builder()
        clf.fit(X_train, y_enc)
        model = ('svm', clf)
    return model, le

def predict_single_level(model_tuple, le, X):
    kind, obj = model_tuple
    if kind == 'const':
        y_pred_enc = np.full(X.shape[0], obj, dtype=int)
    else:
        y_pred_enc = obj.predict(X)
    y_pred = pd.Series(le.inverse_transform(y_pred_enc)).replace("NA_LBL", np.nan)
    return y_pred

def eval_metrics(y_true, y_pred) -> Dict[str, float]:
    yt = y_true.astype(str)
    yp = y_pred.astype(str)
    return {
        'accuracy': accuracy_score(yt, yp),
        'f1_macro': f1_score(yt, yp, average='macro', zero_division=0),
        'f1_weighted': f1_score(yt, yp, average='weighted', zero_division=0),
        'n': len(yt)
    }

def combine_all():
    df_mwpd = standardize_df(read_csv_flex(FILE_MWPD), "MWPD_FULL")
    df_corr = standardize_df(read_csv_flex(FILE_CORRECT), "CORRECTLY_MATCHED")
    df_prod = standardize_df(read_csv_flex(FILE_PRODUCT_MAP), "PRODUCT_GPC_MAPPING")
    df_vali = standardize_df(read_csv_flex(FILE_VALIDATED), "VALIDATED_TEST")
    print("Loaded rows:")
    print(f"  MWPD_FULL                 : {len(df_mwpd):6d}")
    print(f"  correctly_matched_mapped  : {len(df_corr):6d}")
    print(f"  product_gpc_mapping       : {len(df_prod):6d}")
    print(f"  validated_actually_labeled: {len(df_vali):6d}")
    df_all = pd.concat([df_mwpd, df_corr, df_prod, df_vali], axis=0, ignore_index=True)
    df_all['text'] = df_all['product_name'].map(preprocess_keep_symbols)
    df_all = df_all[~df_all['text'].isna() & (df_all['text'].str.strip()!='')]
    for col in hierarchy:
        df_all[col] = df_all[col].map(lambda x: x.strip() if isinstance(x,str) else x)
        df_all[col] = df_all[col].replace("", np.nan).replace("nan", np.nan)
    before = len(df_all)
    df_all['dedup_key'] = df_all['text'].map(make_dedup_key)
    df_all = df_all.drop_duplicates(subset=['text','segment','family','class','brick'], keep='first')
    after = len(df_all)
    df_all.to_csv("all_data_0.85.csv")
    print(f"\nCombined rows before dedup: {before:,} | after dedup: {after:,}")
    return df_all.reset_index(drop=True)

def split_by_key(df_all: pd.DataFrame, seed=RANDOM_SEED) -> pd.DataFrame:
    keys = df_all['dedup_key'].unique().tolist()
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(keys))
    keys = np.array(keys)[perm]
    n = len(keys)
    n_train = int(round(TRAIN_FRAC * n))
    n_val = int(round(VAL_FRAC * n))
    n_train = max(1, min(n_train, n - 2))
    n_val   = max(1, min(n_val, n - n_train - 1))
    train_keys = set(keys[:n_train])
    val_keys   = set(keys[n_train:n_train+n_val])
    test_keys  = set(keys[n_train+n_val:])
    df_all['split'] = np.where(df_all['dedup_key'].isin(train_keys), 'train',
                        np.where(df_all['dedup_key'].isin(val_keys), 'val', 'test'))
    assert train_keys.isdisjoint(val_keys) and train_keys.isdisjoint(test_keys) and val_keys.isdisjoint(test_keys)
    return df_all

def run():
    df_all = combine_all()
    df_all = split_by_key(df_all, seed=RANDOM_SEED)
    print("\nSplit sizes (rows):")
    print(df_all['split'].value_counts(dropna=False).to_string())
    vec = vec_builder()
    X_train_texts = df_all.loc[df_all['split']=='train','text'].tolist()
    vec.fit(X_train_texts)
    models: Dict[str, tuple] = {}
    encoders: Dict[str, LabelEncoder] = {}
    val_metrics_all: Dict[str, Dict[str,float]] = {}
    test_metrics_all: Dict[str, Dict[str,float]] = {}
    for layer in hierarchy:
        tr = df_all[df_all['split']=='train']
        if layer == 'brick':
            tr = tr[~tr['source'].isin(EXCLUDE_SOURCES_FOR_BRICK_TRAIN)]
        va = df_all[df_all['split']=='val']
        te = df_all[df_all['split']=='test']
        tr = tr[~tr[layer].isna()]
        va = va[~va[layer].isna()]
        te = te[~te[layer].isna()]
        print(f"\n[{layer.upper()}] Train/Val/Test sizes (rows): {len(tr):,} / {len(va):,} / {len(te):,}")
        X_tr = vec.transform(tr['text'])
        X_va = vec.transform(va['text'])
        X_te = vec.transform(te['text'])
        model, le = train_single_level(X_tr, tr[layer])
        models[layer] = model
        encoders[layer] = le
        y_va_pred = predict_single_level(model, le, X_va)
        y_te_pred = predict_single_level(model, le, X_te)
        val_metrics = eval_metrics(va[layer], y_va_pred)
        test_metrics = eval_metrics(te[layer], y_te_pred)
        val_metrics_all[layer] = val_metrics
        test_metrics_all[layer] = test_metrics
        print(f"  Val  | n={val_metrics['n']:5d} | Acc={val_metrics['accuracy']:.4f} | F1_macro={val_metrics['f1_macro']:.4f} | F1_weighted={val_metrics['f1_weighted']:.4f}")
        print(f"  Test | n={test_metrics['n']:5d} | Acc={test_metrics['accuracy']:.4f} | F1_macro={test_metrics['f1_macro']:.4f} | F1_weighted={test_metrics['f1_weighted']:.4f}")
    def avg(metrics: Dict[str, Dict[str,float]]):
        acc = np.mean([metrics[l]['accuracy'] for l in hierarchy])
        f1m = np.mean([metrics[l]['f1_macro'] for l in hierarchy])
        f1w = np.mean([metrics[l]['f1_weighted'] for l in hierarchy])
        return acc, f1m, f1w
    v_acc, v_f1m, v_f1w = avg(val_metrics_all)
    t_acc, t_f1m, t_f1w = avg(test_metrics_all)
    print("\nAverage metrics across layers:")
    print(f"  Validation | Acc={v_acc:.4f} | F1_macro={v_f1m:.4f} | F1_weighted={v_f1w:.4f}")
    print(f"  Test       | Acc={t_acc:.4f} | F1_macro={t_f1m:.4f} | F1_weighted={t_f1w:.4f}")
    te_full = df_all[df_all['split']=='test'].copy().dropna(subset=hierarchy)
    if len(te_full):
        X_te_full = vec.transform(te_full['text'])
        preds = {}
        for layer in hierarchy:
            preds[layer] = predict_single_level(models[layer], encoders[layer], X_te_full).values
        preds_df = pd.DataFrame(preds)
        y_true_df = te_full[hierarchy].reset_index(drop=True)
        def build_prediction_output_df(item_name_series, y_true_df, y_pred_df, item_name_col='item_name'):
            out_cols = {item_name_col: item_name_series}
            level_correct_cols = []
            for lvl in hierarchy:
                truth_col = f'{lvl}_truth'
                pred_col = f'{lvl}_pred'
                corr_col = f'{lvl}_correct'
                out_cols[truth_col] = y_true_df[lvl]
                out_cols[pred_col] = y_pred_df[lvl]
                correct = (y_true_df[lvl].astype(str) == y_pred_df[lvl].astype(str)).astype(int)
                out_cols[corr_col] = correct
                level_correct_cols.append(corr_col)
            df_out = pd.DataFrame(out_cols)
            df_out['all_levels_correct'] = (df_out[level_correct_cols].sum(axis=1) == len(hierarchy)).astype(int)
            return df_out
        item_name_series = te_full['product_name'].reset_index(drop=True)
        prediction_output_df = build_prediction_output_df(
            item_name_series=item_name_series,
            y_true_df=y_true_df,
            y_pred_df=preds_df.reset_index(drop=True),
            item_name_col='item_name'
        )
        os.makedirs('data', exist_ok=True)
        prediction_output_df.to_csv('data/prediction_output.csv', index=False)
        print("\nSaved detailed test predictions to data/prediction_output.csv")
    else:
        print("\n[INFO] No complete test rows with all labels to export detailed predictions.")

if __name__ == "__main__":
    run()