"""
AuraSync — EDA ve Veri Ön İşleme (Bölüm 0)
============================================
Amaç: Rubrikte "ayrı, sunulabilir bir EDA bölümü" olarak aranan kısmı
üretmek — eksik veri oranları, sayısal/kategorik dağılım grafikleri,
kategori dengesizliği ve daha önce debugging sırasında bulunan iki veri
kalitesi sorununun (article_id zfill, arl_rules formatı) regresyon testleri.

Çalıştırma (Mac'te, Downloads klasöründeyken):
    python eda_preprocessing.py

Farklı bir veri klasörü kullanıyorsan:
    AURASYNC_DATA_DIR=/baska/yol python eda_preprocessing.py

Beklenen dosyalar (DATA_DIR içinde; bulunamayan dosya atlanır, script
hata vermez):
    articles.csv, customers.csv, transactions_train.csv   (H&M)
    mercari veri dosyası                                   (FILES sözlüğünde adını güncelle)
    clothing-fit / ModCloth-RentTheRunway dosyası           (FILES sözlüğünde adını güncelle)
    customer_features.csv                                   (Economist churn etiketi, varsa)
    arl_rules.csv                                           (format regresyon testi için)

Çıktılar:
    <DATA_DIR>/eda_outputs/*.png
    <DATA_DIR>/eda_outputs/eda_summary.json
"""

# %% [markdown]
# ## 0. Exploratory Data Analysis (EDA) ve Veri Ön İşleme
#
# Bu bölüm dört ajanın da beslendiği ham veri setlerini modelleme öncesinde
# inceler: eksik veri oranları, sayısal/kategorik dağılımlar, kategori
# dengesizliği ve daha önce debugging sırasında tespit edilen iki veri
# kalitesi sorununun regresyon testleri.

# %%
import os
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
sns.set_style("whitegrid")

DATA_DIR = os.environ.get("AURASYNC_DATA_DIR", "/Users/mac/Downloads")
OUT_DIR = os.path.join(DATA_DIR, "eda_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Dosya adlarını kendi ortamına göre güncelle
FILES = {
    "articles": "articles.csv",
    "customers": "customers.csv",
    "transactions": "transactions_train.csv",
    "mercari": "train.tsv",            # <-- güncelle
    "clothing_fit": "modcloth_final_data.json",   # <-- güncelle (RentTheRunway/ModCloth)
    "customer_features": "customer_features.csv",
}

summary = {"missing_data": {}, "shape": {}, "notes": []}


def save_and_show(fig, filename, title=None):
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.show()


def try_load(key_name, **kwargs):
    file_name = FILES.get(key_name)
    if not file_name:
        return None
        
    path = os.path.join(DATA_DIR, file_name)
    if not os.path.exists(path):
        print(f"[bulunamadı] {key_name} at {path}")
        return None
        
    # Check the extension to use the correct loader
    if file_name.endswith('.json'):
        df = pd.read_json(path, lines=True, **kwargs)
    elif file_name.endswith('.tsv'):
        df = pd.read_csv(path, sep='\t', **kwargs)
    else:
        df = pd.read_csv(path, **kwargs)
        
    print(f"[yüklendi] {key_name}: {df.shape}")
    return df


# %% [markdown]
# ### 1. Ham veri setlerinin yüklenmesi

# %%
articles = try_load("articles", dtype={"article_id": str})
customers = try_load("customers")
transactions = try_load("transactions", dtype={"article_id": str}, parse_dates=["t_dat"])
mercari = try_load("mercari", engine="python")
clothing_fit = try_load("clothing_fit")
customer_features = try_load("customer_features")

datasets = {
    "articles (H&M)": articles,
    "customers (H&M)": customers,
    "transactions (H&M)": transactions,
    "mercari": mercari,
    "clothing_fit": clothing_fit,
    "customer_features": customer_features,
}
datasets = {k: v for k, v in datasets.items() if v is not None}

# %% [markdown]
# ### 2. Eksik veri oranları (%)

# %%
for name, df in datasets.items():
    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    summary["missing_data"][name] = miss.round(2).to_dict()
    print(f"\n--- {name} ---")
    print(miss if len(miss) else "Eksik veri yok.")

if datasets:
    fig, ax = plt.subplots(figsize=(10, 5))
    for name, df in datasets.items():
        miss = df.isna().mean() * 100
        miss = miss[miss > 0].sort_values(ascending=False)
        if len(miss):
            ax.barh([f"{name}: {c}" for c in miss.index], miss.values)
    ax.set_xlabel("Eksik oran (%)")
    save_and_show(fig, "missing_data_ratios.png", "Eksik Veri Oranları")

# %% [markdown]
# ### 3. Veri kalitesi regresyon testleri
# Debugging sırasında bulunan iki sorunun burada otomatik kontrolü:
# (a) `article_id` baştaki sıfır kaybı, (b) `arl_rules.csv` formatı.

# %%
if articles is not None:
    bad_len = (articles["article_id"].astype(str).str.len() != 10).sum()
    print(f"article_id uzunluğu != 10 olan kayıt: {bad_len} (beklenen: 0)")
    summary["notes"].append(f"article_id zfill(10) kontrolü: {bad_len} hatalı kayıt")

arl_path = os.path.join(DATA_DIR, "arl_rules.csv")
if os.path.exists(arl_path):
    arl = pd.read_csv(arl_path)
    looks_like_frozenset = arl.iloc[:, 0].astype(str).str.contains("frozenset").any()
    print(f"arl_rules.csv frozenset repr içeriyor mu: {looks_like_frozenset} (beklenen: False)")
    summary["notes"].append(f"arl_rules.csv frozenset format kontrolü: {looks_like_frozenset}")
else:
    print("[atlandı] arl_rules.csv bulunamadı")

# %% [markdown]
# ### 4. Sayısal dağılımlar

# %%
if customers is not None and "age" in customers:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(customers["age"].dropna(), bins=40, ax=ax)
    save_and_show(fig, "age_distribution.png", "Müşteri Yaş Dağılımı")

if transactions is not None and "price" in transactions:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(np.log1p(transactions["price"]), bins=50, ax=ax)
    ax.set_xlabel("log1p(price)")
    save_and_show(fig, "transaction_price_distribution.png", "İşlem Fiyatı Dağılımı")

if mercari is not None and "price" in mercari:
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(np.log1p(mercari["price"]), bins=50, ax=ax)
    ax.set_xlabel("log1p(price)")
    save_and_show(fig, "mercari_price_distribution.png", "Mercari Fiyat Dağılımı")

# %% [markdown]
# ### 5. Kategori dengesizliği

# %%
def plot_top_categories(series, title, filename, top_n=20):
    fig, ax = plt.subplots(figsize=(10, 5))
    series.value_counts().head(top_n).plot(kind="bar", ax=ax)
    ax.set_ylabel("Adet")
    save_and_show(fig, filename, title)


if articles is not None:
    if "product_type_name" in articles:
        plot_top_categories(articles["product_type_name"], "product_type_name Dağılımı (Top 20)", "category_product_type.png")
    if "index_group_name" in articles:
        plot_top_categories(articles["index_group_name"], "index_group_name Dağılımı", "category_index_group.png")
    if "garment_group_name" in articles:
        plot_top_categories(articles["garment_group_name"], "garment_group_name Dağılımı", "category_garment_group.png")

if mercari is not None and "category_name" in mercari:
    plot_top_categories(mercari["category_name"], "Mercari category_name Dağılımı (Top 20)", "mercari_category.png")

if transactions is not None and "sales_channel_id" in transactions:
    plot_top_categories(transactions["sales_channel_id"].astype(str), "sales_channel_id Dağılımı", "sales_channel.png")

# %% [markdown]
# ### 6. Hedef değişken dengesizliği (modelleme kararları için kritik)
# Not: Kolon adlarını kendi customer_features.csv / clothing_fit şemana göre kontrol et.

# %%
if clothing_fit is not None and "fit" in clothing_fit:
    plot_top_categories(clothing_fit["fit"], "Stylist — fit Etiketi Dengesi", "target_fit_balance.png")

if customer_features is not None:
    churn_cols = [c for c in customer_features.columns if "churn" in c.lower()]
    if churn_cols:
        plot_top_categories(customer_features[churn_cols[0]].astype(str), f"Economist — {churn_cols[0]} Dengesi", "target_churn_balance.png")
    else:
        print("customer_features.csv içinde 'churn' geçen bir kolon bulunamadı — kolon adını kontrol et.")

# %% [markdown]
# ### 7. Zaman serisi — aylık işlem hacmi

# %%
if transactions is not None and "t_dat" in transactions:
    monthly = transactions.groupby(transactions["t_dat"].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly.plot(kind="line", marker="o", ax=ax)
    ax.set_ylabel("İşlem sayısı")
    save_and_show(fig, "monthly_transaction_volume.png", "Aylık İşlem Hacmi")

# %% [markdown]
# ### 8. Özet

# %%
with open(os.path.join(OUT_DIR, "eda_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print("\n=== EDA TAMAMLANDI ===")
print(f"Grafikler ve özet: {OUT_DIR}")
for note in summary["notes"]:
    print(f"- {note}")
