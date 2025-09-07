import torch
import torch.nn.functional as F
import pandas as pd
import chardet
import unicodedata
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import re, warnings, os
from typing import List, Optional, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.svm import LinearSVC
warnings.filterwarnings('ignore')
from teradataml import *
from teradataml.dataframe.copy_to import copy_to_sql
import re
import json
from typing import List

FILE_MWPD = 'data/MWPD_FULL.csv'
FILE_CORRECT = 'data/correctly_matched_mapped_gpc.csv'
FILE_PRODUCT_MAP = 'data/product_gpc_mapping.csv'
FILE_VALIDATED = 'data/validated_actually_labeled_test_dataset.csv'

hierarchy = ['segment','family','class','brick']


from constants import ALL_STOPWORDS, ALL_BRANDS, GPC_PATH, PROMPT_PATH, JIO_MART_DATASET_MAPPED
from modules.models import (
    SentenceEmbeddingModel, 
    SentenceEmbeddingConfig,
    OpusTranslationModel,
    OpusTranslationModelConfig,
    LLMModel, 
    LLMModelConfig,
    TfidfClassifier,
    TfidfClassifierConfig,
    HierarchicalGPCClassifier,
    LogisticRegressionConfig,
    WeightedLogisticRegressionClassifier
)

def remove_repeated_words(text):
    text = text.split()
    final_text = []
    for word in text:
        if word in final_text:
            continue
        final_text.append(word)

    return " ".join(final_text)

def remove_brand_name(text: str) -> str:
    for brand in ALL_BRANDS:
        if brand in text:
            text = text.replace(brand, "")
            break

    return text

def remove_strings(text: str, strings: List[str]) -> str:
    for s in strings:
        s = str(s)
        if s in text:
            text = text.replace(s, "")

    return text

def remove_numbers(text: str, remove_string: bool = False) -> str:
    text = text.split()
    text = [t for t in text if not re.search(r"\d", t)] if remove_string else [re.sub(r"\d+", "", t) for t in text]

    return " ".join(text)

def remove_stopwords(text: str):
    text = text.split()
    text = [t for t in text if t not in ALL_STOPWORDS or t == "can"]

    return " ".join(text)

def remove_punctuations(text: str) -> str:
    text = re.sub(r"[^\w\s]", " ", text)

    return " ".join(text.strip().split())

def remove_special_chars(text: str) -> str:
    text = re.sub(r"[-_/\\|]", " ", text)  

    return " ".join(text.strip().split()).lower()

def remove_extra_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def clean_text(text) -> str:
    # text = row["Item_Name"]
    # brand = row["Brand"]
    # text = remove_strings(text, [brand])
    text = remove_punctuations(text)
    text = remove_numbers(text)
    text = remove_extra_space(text)
    # text = remove_brand_name(text)
    # text = remove_stopwords(text)
    # if unit not in text and text == "":
        # text += unit

    return text.lower()

def load_embedding_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = SentenceEmbeddingConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = SentenceEmbeddingModel(config)

    return model


def load_tfidf_random_forest():
    config = TfidfClassifierConfig()
    model = TfidfClassifier(config) 

    return model

def load_logistic_regressiong():
    special_weights = {"food beverage": 5.0}
    config = LogisticRegressionConfig()
    model = WeightedLogisticRegressionClassifier(
        config=config, 
        special_class_weights=special_weights,
        default_weight=1.0
    )
    return model

def load_logistic_regression_balanced():
    config = LogisticRegressionConfig()
    model = WeightedLogisticRegressionClassifier(
        config=config,
        use_balanced=True  
    )
    return model

def load_llm_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = LLMModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = LLMModel(config, PROMPT_PATH)

    return model

def load_HierarchicalGPCClassifier(config_path: str, gpc_data_df):
    with open(config_path, "r") as f:
        config_dict = json.load(f)
    
    try:
        config = LLMModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")

    model = HierarchicalGPCClassifier(config, PROMPT_PATH, gpc_data_df)

    return model

def load_translation_model(config_path: str):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    try:
        config = OpusTranslationModelConfig(**config_dict)
    except TypeError as e:
        raise ValueError(f"Invalid configuration keys: {e}.")
    
    model = OpusTranslationModel(config)

    return model

def join_non_empty(*args):
    return " ".join([str(a).strip() for a in args if pd.notna(a) and str(a).strip()])
    

def load_gpc_to_classes():
    df = pd.read_excel(GPC_PATH)

    df["class_name"] = (
        df["SegmentTitle"].fillna("") + " " +
        df["FamilyTitle"] + " " + 
        df["ClassTitle"] + " " +
        df["BrickTitle"].fillna("")
    )

    df["class_name_cleaned"] = df["class_name"].apply(remove_repeated_words)

    df["description"] = df.apply(lambda row: join_non_empty(
        row["BrickDefinition_Includes"],
        row["BrickDefinition_Excludes"],
    ), axis=1)

    df_new = df[["class_name", "class_name_cleaned", "description"]]

    return df_new

def cluster_topk_classes(cluster_embeddings: List[List[float]], classes_embeddings: List[List[float]], k: int) -> torch.Tensor:
    cluster_embeddings = F.normalize(cluster_embeddings, p=2, dim=1)
    classes_embeddings = F.normalize(classes_embeddings, p=2, dim=1)

    scores = (cluster_embeddings @ classes_embeddings.T)

    topk_classes = torch.topk(scores, k=k)

    return topk_classes[1]
 
def detect_file_encoding(file_path, n_bytes=100000):
    with open(file_path, "rb") as f:
        raw_data = f.read(n_bytes)
        result = chardet.detect(raw_data)
    return result

def unicode_clean(s):
    if not isinstance(s, str):
        return s
    s = unicodedata.normalize('NFKC', s)
    s = ''.join(c for c in s if unicodedata.category(c)[0] != 'C') 
    return s.strip()

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
    segment_candidates = ["segment","segmenttitle","segment_title","predicted_segment", "Segment", "SegmentTitle"]
    family_candidates  = ["family","familytitle","family_title","predicted_family", "Family", "FamilyTitle"]
    class_candidates   = ["class","classtitle","class_title","predicted_class","categoryclass", "Class", "ClassTitle"]
    brick_candidates   = ["brick","bricktitle","brick_title","predicted_brick","gpc_brick", "Brick", "BrickTitle"]
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
    out['segment'] = df[scol].astype(str) if scol is not None else pd.Series([np.nan]*len(df))
    out['family']  = df[fcol].astype(str) if fcol is not None else pd.Series([np.nan]*len(df))
    out['class']   = df[ccol].astype(str) if ccol else np.nan
    out['brick']   = df[bcol].astype(str) if bcol else np.nan
    for col in ['product_name','description','segment','family','class','brick']:
        out[col] = out[col].astype(str).map(lambda x: re.sub(r"\s+"," ",x).strip() if isinstance(x,str) else x)
        out[col] = out[col].replace("", np.nan).replace("nan", np.nan)
    for col in ['segment','family','class','brick']:
        out[col] = out[col].map(standardize_label)
    out = out.dropna(subset=["segment"])
    out['source'] = source_name
    out = out[~out['product_name'].isna() & (out['product_name'].str.strip()!="")].reset_index(drop=True)
    return out


def make_dedup_key(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[\W_]+", " ", s)
    s = re.sub(r"\s+"," ", s).strip()
    return s

def combine_all():
    df_mwpd = standardize_df(read_csv_flex(FILE_MWPD), "MWPD_FULL")
    df_corr = standardize_df(read_csv_flex(FILE_CORRECT), "CORRECTLY_MATCHED")
    df_prod = standardize_df(read_csv_flex(FILE_PRODUCT_MAP), "PRODUCT_GPC_MAPPING")
    df_vali = standardize_df(read_csv_flex(FILE_VALIDATED), "VALIDATED_TEST")
    df_jio_mart = standardize_df(read_csv_flex(JIO_MART_DATASET_MAPPED), "JIO_MART")
    print("Loaded rows:")
    print(f"  MWPD_FULL                 : {len(df_mwpd):6d}")
    print(f"  correctly_matched_mapped  : {len(df_corr):6d}")
    print(f"  product_gpc_mapping       : {len(df_prod):6d}")
    print(f"  validated_actually_labeled: {len(df_vali):6d}")
    print(f"  Jio_Mart Dataset: {len(df_jio_mart):6d}")
    df_all = pd.concat([df_mwpd, df_corr, df_prod, df_vali, df_jio_mart], axis=0, ignore_index=True)
    df_all['text'] = df_all['product_name'].map(preprocess_keep_symbols)
    df_all = df_all[~df_all['text'].isna() & (df_all['text'].str.strip()!='')]
    for col in hierarchy:
        df_all[col] = df_all[col].map(lambda x: x.strip() if isinstance(x,str) else x)
        df_all[col] = df_all[col].replace("", np.nan).replace("nan", np.nan)
    df_all['dedup_key'] = df_all['text'].map(make_dedup_key)
    #df_all = df_all.drop_duplicates(subset=['text','segment','family','class','brick'], keep='first')
    #print(df_all.groupby("source")[["segment","family"]].apply(lambda x: x.isna().mean()))
    df_all.to_csv("all_data_0.85.csv")
    return df_all.reset_index(drop=True)