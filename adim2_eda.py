"""
=============================================================
ADIM 2: KEŞİFSEL VERİ ANALİZİ (EDA)
=============================================================
Bu script:
- 3 bankanın verisini yükler ve birleştirir
- Temel istatistikleri çıkarır
- Çözülme oranı, memnuniyet dağılımı, zaman trendi,
  şikayet uzunluğu, en sık anahtar kelimeler gibi
  görseller üretir → results/figures/ klasörüne kaydeder

ÇALIŞTIRMA: python adim2_eda.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"   # Türkçe karakter desteği

# ----------------------------------------------------------
# 0. KLASÖRLER
# ----------------------------------------------------------
os.makedirs("results/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ----------------------------------------------------------
# 1. VERİ YÜKLEME
# ----------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DOSYALAR = {
    "VakifBank":   os.path.join(BASE_DIR, "data", "raw", "sikayetvar_vakifbank.csv"),
    "IsBank":      os.path.join(BASE_DIR, "data", "raw", "sikayetvar_isbank.csv"),
    "KuveytTurk":  os.path.join(BASE_DIR, "data", "raw", "sikayetvar_kuveyt_turk.csv"),
}

BANKA_ADI = {
    "VakifBank":  "VakıfBank",
    "IsBank":     "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}

RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBank":     "#B71C1C",
    "KuveytTurk": "#1B5E20",
}

dfler = []
for anahtar, dosya in DOSYALAR.items():
    df = pd.read_csv(dosya, encoding="utf-8")
    df["banka_key"] = anahtar
    df["banka"] = BANKA_ADI[anahtar]
    dfler.append(df)

veri = pd.concat(dfler, ignore_index=True)
print(f"Toplam kayıt: {len(veri)}")

# Tarih sütununu düzenle
veri["date"] = pd.to_datetime(veri["date"], errors="coerce")
veri["ay"] = veri["date"].dt.to_period("M")
veri["ay_str"] = veri["date"].dt.strftime("%Y-%m")

# Metin uzunluğu
veri["metin_uzunluk"] = veri["full_text"].fillna("").apply(len)
veri["kelime_sayisi"] = veri["full_text"].fillna("").apply(lambda x: len(x.split()))

print("\n--- GENEL İSTATİSTİKLER ---")
print(veri.groupby("banka")[["metin_uzunluk", "kelime_sayisi", "satisfaction", "view_count"]].mean().round(2))

# ----------------------------------------------------------
# 2. GRAFİK 1: Banka başına şikayet sayısı
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sayilar = veri["banka"].value_counts()
renkler = [RENKLER[k] for k in ["VakifBank", "IsBank", "KuveytTurk"]
           if BANKA_ADI[k] in sayilar.index]
bars = ax.bar(sayilar.index, sayilar.values,
              color=[RENKLER[k] for k in DOSYALAR if BANKA_ADI[k] in sayilar.index])
for bar, val in zip(bars, sayilar.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            str(val), ha="center", fontsize=12, fontweight="bold")
ax.set_title("Bankaya Göre Şikayet Sayısı", fontsize=14, fontweight="bold")
ax.set_ylabel("Şikayet Sayısı")
ax.set_xlabel("")
plt.tight_layout()
plt.savefig("results/figures/01_sikayet_sayisi.png", dpi=150)
plt.close()
print("\n✓ Grafik 1 kaydedildi: 01_sikayet_sayisi.png")

# ----------------------------------------------------------
# 3. GRAFİK 2: Çözülme oranı (banka bazlı)
# ----------------------------------------------------------
cozulme = veri.groupby(["banka", "is_resolved"]).size().unstack(fill_value=0)
cozulme_oran = cozulme.div(cozulme.sum(axis=1), axis=0) * 100

fig, ax = plt.subplots(figsize=(9, 5))
cozulme_oran.plot(kind="bar", ax=ax,
                  color=["#43A047", "#E53935"],
                  edgecolor="white", width=0.6)
ax.set_title("Bankaya Göre Çözülme Oranı (%)", fontsize=14, fontweight="bold")
ax.set_ylabel("Oran (%)")
ax.set_xlabel("")
ax.legend(title="Durum", loc="upper right")
ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=11)
for container in ax.containers:
    ax.bar_label(container, fmt="%.1f%%", fontsize=9, padding=2)
plt.tight_layout()
plt.savefig("results/figures/02_cozulme_orani.png", dpi=150)
plt.close()
print("✓ Grafik 2 kaydedildi: 02_cozulme_orani.png")

# ----------------------------------------------------------
# 4. GRAFİK 3: Memnuniyet skoru dağılımı
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
bankalar = ["VakifBank", "IsBank", "KuveytTurk"]

for i, (ax, banka_key) in enumerate(zip(axes, bankalar)):
    banka_adi = BANKA_ADI[banka_key]
    alt = veri[veri["banka"] == banka_adi]["satisfaction"].dropna()
    counts = alt.value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values,
           color=RENKLER[banka_key], edgecolor="white")
    ax.set_title(banka_adi, fontsize=12, fontweight="bold")
    ax.set_xlabel("Memnuniyet Skoru (1-5)")
    if i == 0:
        ax.set_ylabel("Şikayet Sayısı")
    ax.set_xticks([1, 2, 3, 4, 5])

fig.suptitle("Memnuniyet Skoru Dağılımı (Bankaya Göre)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/03_memnuniyet_dagilimi.png", dpi=150)
plt.close()
print("✓ Grafik 3 kaydedildi: 03_memnuniyet_dagilimi.png")

# ----------------------------------------------------------
# 5. GRAFİK 4: Aylık şikayet trendi
# ----------------------------------------------------------
aylik = veri.groupby(["ay_str", "banka"]).size().reset_index(name="sayi")
aylik_pivot = aylik.pivot(index="ay_str", columns="banka", values="sayi").fillna(0)
aylik_pivot = aylik_pivot.sort_index()

fig, ax = plt.subplots(figsize=(12, 5))
for banka_key, banka_adi in BANKA_ADI.items():
    if banka_adi in aylik_pivot.columns:
        ax.plot(aylik_pivot.index, aylik_pivot[banka_adi],
                marker="o", linewidth=2, markersize=4,
                color=RENKLER[banka_key], label=banka_adi)

ax.set_title("Aylık Şikayet Trendi", fontsize=14, fontweight="bold")
ax.set_xlabel("Ay")
ax.set_ylabel("Şikayet Sayısı")
ax.legend()
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("results/figures/04_aylik_trend.png", dpi=150)
plt.close()
print("✓ Grafik 4 kaydedildi: 04_aylik_trend.png")

# ----------------------------------------------------------
# 6. GRAFİK 5: Şikayet metni uzunluğu (kelime sayısı)
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
for banka_key, banka_adi in BANKA_ADI.items():
    alt = veri[veri["banka"] == banka_adi]["kelime_sayisi"]
    alt = alt[alt < alt.quantile(0.95)]   # aykırı değerleri kırp
    ax.hist(alt, bins=40, alpha=0.6,
            color=RENKLER[banka_key], label=banka_adi, edgecolor="none")

ax.set_title("Şikayet Metni Kelime Sayısı Dağılımı", fontsize=14, fontweight="bold")
ax.set_xlabel("Kelime Sayısı")
ax.set_ylabel("Frekans")
ax.legend()
plt.tight_layout()
plt.savefig("results/figures/05_kelime_sayisi.png", dpi=150)
plt.close()
print("✓ Grafik 5 kaydedildi: 05_kelime_sayisi.png")

# ----------------------------------------------------------
# 7. GRAFİK 6: En sık anahtar kelimeler (keywords sütunu)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

for i, (banka_key, banka_adi) in enumerate(BANKA_ADI.items()):
    ax = axes[i]
    keywords = veri[veri["banka"] == banka_adi]["keywords"].dropna()
    tum_kelimeler = []
    for kw in keywords:
        tum_kelimeler.extend([k.strip().lower() for k in str(kw).split(",") if k.strip()])
    en_sik = Counter(tum_kelimeler).most_common(15)
    kelimeler, frekanslar = zip(*en_sik) if en_sik else ([], [])
    ax.barh(list(reversed(kelimeler)), list(reversed(frekanslar)),
            color=RENKLER[banka_key], edgecolor="none")
    ax.set_title(banka_adi, fontsize=12, fontweight="bold")
    ax.set_xlabel("Frekans")

fig.suptitle("En Sık Şikayet Konuları (Anahtar Kelimeler)", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/06_anahtar_kelimeler.png", dpi=150)
plt.close()
print("✓ Grafik 6 kaydedildi: 06_anahtar_kelimeler.png")

# ----------------------------------------------------------
# 8. GRAFİK 7: Görüntülenme sayısı kutu grafiği
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
banka_listesi = [BANKA_ADI[k] for k in bankalar]
data_boxplot = [veri[veri["banka"] == b]["view_count"].dropna().clip(upper=5000).values
                for b in banka_listesi]
bp = ax.boxplot(data_boxplot, labels=banka_listesi, patch_artist=True,
                medianprops=dict(color="white", linewidth=2))
for patch, banka_key in zip(bp["boxes"], bankalar):
    patch.set_facecolor(RENKLER[banka_key])
ax.set_title("Şikayet Görüntülenme Sayısı Dağılımı (Aykırı değerler 5000'de kırpıldı)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Görüntülenme Sayısı")
plt.tight_layout()
plt.savefig("results/figures/07_goruntulenme.png", dpi=150)
plt.close()
print("✓ Grafik 7 kaydedildi: 07_goruntulenme.png")

# ----------------------------------------------------------
# 9. ÖZET TABLO - CSV olarak kaydet
# ----------------------------------------------------------
ozet = veri.groupby("banka").agg(
    sikayet_sayisi=("id", "count"),
    cozulme_orani=("is_resolved", lambda x: (x == "Çözüldü").mean() * 100),
    ort_memnuniyet=("satisfaction", "mean"),
    ort_kelime_sayisi=("kelime_sayisi", "mean"),
    ort_goruntulenme=("view_count", "mean"),
    toplam_goruntulenme=("view_count", "sum"),
).round(2)

ozet.to_csv("data/processed/ozet_istatistikler.csv", encoding="utf-8")
print("\n✓ Özet istatistikler kaydedildi: data/processed/ozet_istatistikler.csv")
print("\n--- ÖZET TABLO ---")
print(ozet.to_string())

# ----------------------------------------------------------
# 10. BİRLEŞİK VERİYİ KAYDET (sonraki adımlar için)
# ----------------------------------------------------------
veri.to_csv("data/processed/veri_ham_birlesmis.csv", index=False, encoding="utf-8")
print("\n✓ Birleşik ham veri kaydedildi: data/processed/veri_ham_birlesmis.csv")

print("\n" + "=" * 60)
print("ADIM 2 TAMAMLANDI!")
print("7 grafik → results/figures/ klasöründe")
print("Sıradaki adım: python adim3_onisleme.py")
print("=" * 60)
