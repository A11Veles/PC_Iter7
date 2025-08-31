import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from teradataml import DataFrame
from src.modules.db import TeradataDatabase
from src.modules.models import SentenceEmbeddingModel, SentenceEmbeddingConfig  # used only for free-text + E5

# ---------- Config ----------
DEVICE = "cuda"
SCHEMA = "demo_user"
PRODUCTS_TBL = "products"       # id, translated_name
CLASSES_TBL  = "classes"        # id, class_name
P_EMB_TBL    = "p_embeddings"   # id, embed_0..embed_N
C_EMB_TBL    = "c_embeddings"   # id, embed_0..embed_N
ACTUALS_TBL  = "actual_classes" # schema may vary; handled flexibly

# ---------- DB helpers ----------
@st.cache_resource
def get_db():
    db = TeradataDatabase()
    db.connect()
    return db

@st.cache_data
def load_data():
    _ = get_db()  # ensure teradataml context

    products_df = DataFrame.from_table(PRODUCTS_TBL, schema_name=SCHEMA)[["id", "translated_name"]].to_pandas()
    classes_df  = DataFrame.from_table(CLASSES_TBL,  schema_name=SCHEMA)[["id", "class_name"]].to_pandas()
    p_emb_df    = DataFrame.from_table(P_EMB_TBL,    schema_name=SCHEMA).to_pandas()
    c_emb_df    = DataFrame.from_table(C_EMB_TBL,    schema_name=SCHEMA).to_pandas()

    # Merge embeddings → names
    product_full = p_emb_df.merge(products_df, on="id", how="left")
    class_full   = c_emb_df.merge(classes_df,  on="id", how="left")

    # --- Ground truth (robust join)
    try:
        actuals_df = DataFrame.from_table(ACTUALS_TBL, schema_name=SCHEMA).to_pandas()
        actuals_df.columns = [c.strip().lower() for c in actuals_df.columns]
        classes_df_norm = classes_df.rename(columns={"id": "class_id"}).copy()
        classes_df_norm.columns = [c.strip().lower() for c in classes_df_norm.columns]

        prod_key = "product_id" if "product_id" in actuals_df.columns else ("id" if "id" in actuals_df.columns else None)
        if prod_key is None:
            product_full["true_class_id"] = np.nan
            product_full["true_class_name"] = np.nan
        else:
            if "class_id" in actuals_df.columns:
                gt = actuals_df[[prod_key, "class_id"]].rename(columns={prod_key: "id"})
                gt = gt.merge(
                    classes_df_norm[["class_id", "class_name"]].rename(columns={"class_name": "true_class_name"}),
                    on="class_id", how="left"
                ).rename(columns={"class_id": "true_class_id"})
                product_full = product_full.merge(gt, on="id", how="left")
            elif "class_name" in actuals_df.columns:
                gt = actuals_df[[prod_key, "class_name"]].rename(columns={prod_key: "id"})
                gt = gt.merge(classes_df_norm, on="class_name", how="left")
                gt = gt.rename(columns={"class_id": "true_class_id", "class_name": "true_class_name"})
                product_full = product_full.merge(gt[["id", "true_class_id", "true_class_name"]], on="id", how="left")
            else:
                product_full["true_class_id"] = np.nan
                product_full["true_class_name"] = np.nan
    except Exception:
        product_full["true_class_id"] = np.nan
        product_full["true_class_name"] = np.nan

    return product_full, class_full, classes_df

# ---------- E5 (DB) ----------
@st.cache_data
def load_embeddings_from_db(product_full: pd.DataFrame, class_full: pd.DataFrame):
    prod_cols = sorted([c for c in product_full.columns if c.startswith("embed_")],
                       key=lambda x: int(x.split("_")[1]))
    cls_cols  = sorted([c for c in class_full.columns if c.startswith("embed_")],
                       key=lambda x: int(x.split("_")[1]))
    if not prod_cols or not cls_cols:
        raise ValueError("Missing embed_* columns in DB tables.")

    prod = torch.tensor(product_full[prod_cols].to_numpy(np.float32, copy=False), dtype=torch.float16, device=DEVICE)
    cls  = torch.tensor(class_full[cls_cols].to_numpy(np.float32, copy=False), dtype=torch.float16, device=DEVICE)
    return F.normalize(prod.float(), p=2, dim=1).half(), F.normalize(cls.float(), p=2, dim=1).half()

def predict_topk_e5(prod_vec: torch.Tensor, cls_mat: torch.Tensor, k: int = 3):
    if prod_vec.dim() == 1:
        prod_vec = prod_vec.unsqueeze(0)
    scores = torch.mm(prod_vec, cls_mat.T)  # cosine (normalized)
    vals, idx = torch.topk(scores, k=min(k, cls_mat.size(0)), dim=1)
    return vals[0].cpu().numpy(), idx[0].cpu().numpy()

# ---------- TF-IDF ----------
@st.cache_resource
def build_tfidf(classes_df: pd.DataFrame):
    # Train on class names
    corpus = classes_df["class_name"].fillna("").astype(str).tolist()
    vec = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=1,
        norm="l2"
    )
    cls_mat = vec.fit_transform(corpus)  # shape [C, D]
    return vec, cls_mat

def predict_topk_tfidf(text: str, vec: TfidfVectorizer, cls_mat, k: int = 3):
    q = vec.transform([text or ""])
    sims = linear_kernel(q, cls_mat)  # [1, C]
    idx = np.argsort(-sims[0])[:k]
    vals = sims[0][idx]
    return vals, idx

# ---------- Optional E5 encoder (only for free-text + E5) ----------
@st.cache_resource
def load_encoder() -> SentenceEmbeddingModel:
    cfg = SentenceEmbeddingConfig(
        device=DEVICE, dtype="float16", model_id="intfloat/e5-large-v2",
        truncate_dim=None, convert_to_numpy=False, convert_to_tensor=True,
        use_prompt=True, prompt_config={"classification": "passage: {text}"},
        model_kwargs={"torch_dtype": "float16"},
    )
    return SentenceEmbeddingModel(cfg)

def e5_embed_texts(texts):
    enc = load_encoder()
    emb = enc.get_embeddings([t if t is not None else "" for t in texts], prompt_name="classification")
    if not torch.is_tensor(emb):
        emb = torch.tensor(emb)
    emb = emb.to(DEVICE).to(torch.float32)
    emb = F.normalize(emb, p=2, dim=1)
    return emb

# ---------- UI ----------
st.title("🧠 Product Classification — E5 or TF-IDF")

with st.spinner("Loading data from Teradata…"):
    product_full, class_full, classes_df = load_data()

# Model choice
model_choice = st.radio("Choose model", ["E5 (DB embeddings)", "TF-IDF"], horizontal=True)

# Prepare model-specific assets
if model_choice == "E5 (DB embeddings)":
    with st.spinner("Preparing E5 embeddings…"):
        prod_emb, cls_emb = load_embeddings_from_db(product_full, class_full)
else:
    with st.spinner("Building TF-IDF on class names…"):
        tfidf_vec, tfidf_cls_mat = build_tfidf(classes_df)

# ---------- Browse & classify DB row ----------
st.subheader("Classify a product from the database")

items_per_page = st.selectbox("Items per page", [5, 10, 25, 50], 1)
total_items = len(product_full)
total_pages = (total_items - 1)//items_per_page + 1
page = st.number_input("Page", 1, total_pages, 1)
s, e = (page-1)*items_per_page, min(page*items_per_page, total_items)
st.dataframe(product_full.iloc[s:e][["id", "translated_name", "true_class_name"]], hide_index=True, use_container_width=True)

chosen_id = st.number_input(
    "Enter product id",
    int(product_full["id"].min()),
    int(product_full["id"].max()),
    int(product_full.iloc[0]["id"])
)
if st.button("🔎 Classify selected product"):
    idx_match = product_full.index[product_full["id"] == chosen_id]
    if len(idx_match) == 0:
        st.error("ID not found.")
    else:
        pidx = int(idx_match[0])

        if model_choice == "E5 (DB embeddings)":
            scores, idxs = predict_topk_e5(prod_emb[pidx], cls_emb, k=3)
        else:
            text = product_full.loc[pidx, "translated_name"]
            scores, idxs = predict_topk_tfidf(str(text), tfidf_vec, tfidf_cls_mat, k=3)

        top1_idx = int(idxs[0])
        # classes come from classes_df (id, class_name)
        pred_name = classes_df["class_name"].iloc[top1_idx]
        pred_id   = int(classes_df["id"].iloc[top1_idx])
        st.success(f"Prediction: {pred_name} (id={pred_id}, score={float(scores[0]):.3f})")

        # ground truth
        true_id   = product_full.loc[pidx, "true_class_id"] if "true_class_id" in product_full.columns else np.nan
        true_name = product_full.loc[pidx, "true_class_name"] if "true_class_name" in product_full.columns else np.nan
        if pd.notna(true_id):
            st.write("✅ Correct" if int(true_id) == pred_id else "❌ Incorrect")
            st.write(f"Ground truth: {true_name} (id={int(true_id)})")
        else:
            st.info("No ground truth for this product.")

        st.caption("Top-3:")
        for r, (sc, ci) in enumerate(zip(scores, idxs), start=1):
            cname = classes_df["class_name"].iloc[int(ci)]
            cid   = int(classes_df["id"].iloc[int(ci)])
            st.write(f"{r}. {cname} (id={cid}) — {float(sc):.3f}")

# ---------- Free-text classify ----------
st.subheader("Classify a custom description")
user_text = st.text_area("Type a product description…", "", height=100)
free_model = st.radio("Model for free-text", ["TF-IDF", "E5 (encode now)"], horizontal=True, key="free_model")

if st.button("✨ Classify text"):
    if not user_text.strip():
        st.warning("Please enter some text.")
    else:
        if free_model == "TF-IDF":
            # ensure tf-idf is built
            if model_choice != "TF-IDF":
                tfidf_vec, tfidf_cls_mat = build_tfidf(classes_df)
            scores, idxs = predict_topk_tfidf(user_text, tfidf_vec, tfidf_cls_mat, k=3)
        else:
            # E5 on-the-fly just for this text
            cls_cols  = sorted([c for c in class_full.columns if c.startswith("embed_")],
                               key=lambda x: int(x.split("_")[1]))
            cls_mat = torch.tensor(class_full[cls_cols].to_numpy(np.float32, copy=False), device=DEVICE)
            cls_mat = F.normalize(cls_mat, p=2, dim=1)
            q = e5_embed_texts([user_text])         # [1, dim], normalized
            scores = torch.mm(q, cls_mat.T).cpu().numpy()[0]
            idxs = np.argsort(-scores)[:3]
            scores = scores[idxs]

        top1_idx = int(idxs[0])
        pred_name = classes_df["class_name"].iloc[top1_idx]
        pred_id   = int(classes_df["id"].iloc[top1_idx])
        st.success(f"Prediction: {pred_name} (id={pred_id}, score={float(scores[0]):.3f})")

        st.caption("Top-3:")
        for r, (sc, ci) in enumerate(zip(scores, idxs), start=1):
            cname = classes_df["class_name"].iloc[int(ci)]
            cid   = int(classes_df["id"].iloc[int(ci)])
            st.write(f"{r}. {cname} (id={cid}) — {float(sc):.3f}")
