
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import re, warnings

from constants import E5_LARGE_INSTRUCT_CONFIG_PATH, GPC_PATH
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

def load_gpc():
    gpc_df = pd.read_excel(GPC_PATH)
    cols_to_keep = [
    "SegmentTitle", "SegmentDefinition", 
    "FamilyTitle", "FamilyDefinition", 
    "ClassTitle", "ClassDefinition", 
    "BrickTitle", "BrickDefinition_Includes"
    ]

    gpc_df = gpc_df[cols_to_keep]

    return gpc_df

def join_data():
    train_df = pd.read_csv('data/correctly_matched_mapped_gpc.csv')
    test2_df = pd.read_csv('data/validated_actually_labeled_test_dataset.csv')
    gpc_df = load_gpc()

    segment_map = (gpc_df["SegmentTitle"].astype(str) + " - " + gpc_df["SegmentDefinition"].astype(str)).to_dict()
    family_map  = (gpc_df["FamilyTitle"].astype(str)  + " - " + gpc_df["FamilyDefinition"].astype(str)).to_dict()
    class_map   = (gpc_df["ClassTitle"].astype(str)   + " - " + gpc_df["ClassDefinition"].astype(str)).to_dict()
    brick_map   = (gpc_df["BrickTitle"].astype(str)   + " - " + gpc_df["BrickDefinition_Includes"].astype(str)).to_dict()

    if "segment" in train_df.columns:
        train_df["segment"] = train_df["segment"].map(segment_map).fillna(train_df["segment"])
    if "family" in train_df.columns:
        train_df["family"] = train_df["family"].map(family_map).fillna(train_df["family"])
    if "class" in train_df.columns:
        train_df["class"] = train_df["class"].map(class_map).fillna(train_df["class"])
    if "brick" in train_df.columns:
        train_df["brick"] = train_df["brick"].map(brick_map).fillna(train_df["brick"])

    if "predicted_segment" in test2_df.columns:
        test2_df["predicted_segment"] = test2_df["predicted_segment"].map(segment_map).fillna(test2_df["predicted_segment"])
    if "predicted_family" in test2_df.columns:
        test2_df["predicted_family"] = test2_df["predicted_family"].map(family_map).fillna(test2_df["predicted_family"])
    if "predicted_class" in test2_df.columns:
        test2_df["predicted_class"] = test2_df["predicted_class"].map(class_map).fillna(test2_df["predicted_class"])
    if "predicted_brick" in test2_df.columns:
        test2_df["predicted_brick"] = test2_df["predicted_brick"].map(brick_map).fillna(test2_df["predicted_brick"])

    return train_df, test2_df

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

def eval_weighted_f1(y_true_df, y_pred_df):
    return float(np.mean([f1_score(y_true_df[l], y_pred_df[l], average='weighted', zero_division=0) for l in hierarchy]))

def main():
    embedding_model = load_embedding_model(E5_LARGE_INSTRUCT_CONFIG_PATH)

    train_df, test2_df = join_data()
    train_df = train_df.copy()
    train_df['processed_name'] = train_df['product_name'].apply(preprocess_keep_symbols)
    tr, va = split_data(train_df, seed=42)

    X_va = va['processed_name']

    y_tr = tr[hierarchy].copy()
    y_va = va[hierarchy].copy()

    models = train_per_level(y_tr, embedding_model)

    val_preds = predict_levels(models, X_va, embedding_model)
    val_f1 = eval_weighted_f1(y_va, val_preds)

    t2 = test2_df.copy()
    t2['processed_name'] = t2['translated_name'].apply(preprocess_keep_symbols)
    X_t2 = t2['processed_name']
    y_t2 = t2[['predicted_segment','predicted_family','predicted_class','predicted_brick']].copy()
    print(y_t2.head())
    y_t2.columns = hierarchy
    p2 = predict_levels(models, X_t2, embedding_model)
    test2_f1 = eval_weighted_f1(y_t2, p2)


    print("\\nRESULTS (E5 Model + description)")
    print(f"Val F1:   {val_f1:.4f}")
    print(f"Test2 F1: {test2_f1:.4f}")

if __name__ == "__main__":
    main()


