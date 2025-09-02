import pandas as pd
import numpy as np
import re, warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

warnings.filterwarnings('ignore')
hierarchy = ['segment','family','class','brick']

def preprocess_keep_symbols(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\+\-/\.]', ' ', text)
    return ' '.join(text.split())

def load_data():
    train_df = pd.read_csv('data/correctly_matched_mapped_gpc.csv')
    test1_df = pd.read_csv('data/product_gpc_mapping.csv')
    test2_df = pd.read_csv('data/validated_actually_labeled_test_dataset.csv')
    return train_df, test1_df, test2_df

def split_data(df, seed=42):
    if 'segment' in df.columns:
        return train_test_split(df, test_size=0.2, random_state=seed, stratify=df['segment'])
    return train_test_split(df, test_size=0.2, random_state=seed)

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

def train_per_level(X_train, y_train):
    models, encs = {}, {}
    for lvl in hierarchy:
        le = LabelEncoder()
        y_enc = le.fit_transform(y_train[lvl])
        if len(np.unique(y_enc)) < 2:
            models[lvl] = ('const', int(y_enc[0]))
        else:
            clf = model_builder()
            clf.fit(X_train, y_enc)
            models[lvl] = ('svm', clf)
        encs[lvl] = le
    return models, encs

def predict_levels(models, encs, X):
    out = {}
    for lvl in hierarchy:
        kind, obj = models[lvl]
        if kind == 'const':
            y_pred_enc = np.full(X.shape[0], obj, dtype=int)
        else:
            y_pred_enc = obj.predict(X)
        out[lvl] = encs[lvl].inverse_transform(y_pred_enc)
    return pd.DataFrame(out)

def eval_accuracy(y_true_df, y_pred_df):
    return {l: accuracy_score(y_true_df[l], y_pred_df[l]) for l in hierarchy}

def run():
    train_df, test1_df, test2_df = load_data()

    train_df = train_df.copy()
    train_df['processed_name'] = train_df['product_name'].apply(preprocess_keep_symbols)
    tr, va = split_data(train_df, seed=42)

    vec = vec_builder()
    X_tr = vec.fit_transform(tr['processed_name'])
    X_va = vec.transform(va['processed_name'])

    y_tr = tr[hierarchy].copy()
    y_va = va[hierarchy].copy()

    models, encs = train_per_level(X_tr, y_tr)

    val_preds = predict_levels(models, encs, X_va)
    val_acc = eval_accuracy(y_va, val_preds)

    t1 = test1_df.copy()
    t1['processed_name'] = t1['Name'].apply(preprocess_keep_symbols)
    X_t1 = vec.transform(t1['processed_name'])
    y_t1 = t1[['SegmentTitle','FamilyTitle','ClassTitle','BrickTitle']].copy()
    y_t1.columns = hierarchy
    p1 = predict_levels(models, encs, X_t1)
    test1_acc = eval_accuracy(y_t1, p1)

    t2 = test2_df.copy()
    t2['processed_name'] = t2['translated_name'].apply(preprocess_keep_symbols)
    X_t2 = vec.transform(t2['processed_name'])
    y_t2 = t2[['predicted_segment','predicted_family','predicted_class','predicted_brick']].copy()
    y_t2.columns = hierarchy
    p2 = predict_levels(models, encs, X_t2)
    test2_acc = eval_accuracy(y_t2, p2)

    print("\nRESULTS (LinearSVC (1,2)-gram, keep symbols)")
    for l in hierarchy:
        print(f"{l.capitalize()} | Val Acc: {val_acc[l]:.4f} | Test1 Acc: {test1_acc[l]:.4f} | Test2 Acc: {test2_acc[l]:.4f}")

if __name__ == "__main__":
    run()
