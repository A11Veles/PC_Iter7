import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import re, warnings

from constants import E5_LARGE_INSTRUCT_CONFIG_PATH
from utils import load_embedding_model

hierarchy = ['segment', 'family', 'class', 'brick']
warnings.filterwarnings('ignore')

def preprocess_keep_symbols(text):
    if pd.isna(text): return ""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s\+\-\.\/]', ' ', text)
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

def train_per_level(y_train, embedding_model):
    models = {}
    for lvl in hierarchy:
        unique_labels = y_train[lvl].unique()
        label_embeddings = embedding_model.get_embeddings(list(unique_labels))
        models[lvl] = {'labels': unique_labels, 'embeddings': label_embeddings}
    return models

def predict_levels(models, X, embedding_model):
    out = {}
    X_embeddings = embedding_model.get_embeddings(X.tolist())
    for lvl in hierarchy:
        labels = models[lvl]['labels']
        label_embeddings = models[lvl]['embeddings']
        scores = embedding_model.calculate_scores(X_embeddings, label_embeddings)
        predicted_indices = scores.argmax(axis=1)
        predicted_labels = [labels[idx] for idx in predicted_indices]
        out[lvl] = predicted_labels
    return pd.DataFrame(out)

def eval_accuracy(y_true_df, y_pred_df):
    return {l: accuracy_score(y_true_df[l], y_pred_df[l]) for l in hierarchy}

def main():
    embedding_model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)

    train_df, test1_df, test2_df = load_data()
    train_df['processed_name'] = train_df['product_name'].apply(preprocess_keep_symbols)
    tr, va = split_data(train_df, seed=42)

    X_va = va['processed_name']
    y_tr = tr[hierarchy].copy()
    y_va = va[hierarchy].copy()

    models = train_per_level(y_tr, embedding_model)

    val_preds = predict_levels(models, X_va, embedding_model)
    val_acc = eval_accuracy(y_va, val_preds)

    t1 = test1_df.copy()
    t1['processed_name'] = t1['Name'].apply(preprocess_keep_symbols)
    X_t1 = t1['processed_name']
    y_t1 = t1[['SegmentTitle','FamilyTitle','ClassTitle','BrickTitle']].copy()
    y_t1.columns = hierarchy
    p1 = predict_levels(models, X_t1, embedding_model)
    test1_acc = eval_accuracy(y_t1, p1)

    t2 = test2_df.copy()
    t2['processed_name'] = t2['translated_name'].apply(preprocess_keep_symbols)
    X_t2 = t2['processed_name']
    y_t2 = t2[['predicted_segment','predicted_family','predicted_class','predicted_brick']].copy()
    y_t2.columns = hierarchy
    p2 = predict_levels(models, X_t2, embedding_model)
    test2_acc = eval_accuracy(y_t2, p2)

    print("\nRESULTS (E5 Model)")
    for l in hierarchy:
        print(f"{l.capitalize()} | Val Acc: {val_acc[l]:.4f} | Test1 Acc: {test1_acc[l]:.4f} | Test2 Acc: {test2_acc[l]:.4f}")

if __name__ == "__main__":
    main()
