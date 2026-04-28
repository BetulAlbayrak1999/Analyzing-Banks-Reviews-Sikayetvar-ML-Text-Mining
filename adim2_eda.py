"""
=============================================================
ADIM 2 (DÜZELTİLMİŞ): KEŞİFSEL VERİ ANALİZİ (EDA)
=============================================================
Duzeltmeler:
- CSV'ler utf-8-sig ile kaydediliyor → Excel Turkce gosterir
- Null deger islemi raporu ayri CSV'ye yaziliyor
- Kelime sayisi hesabi aciklamali

CALISTIRMA: python adim2_eda.py
=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from collections import Counter
import warnings, os

warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

# ----------------------------------------------------------
# 0. KLASORLER
# ----------------------------------------------------------
os.makedirs("results/figures", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ----------------------------------------------------------
# 1. VERI YUKLEME
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOSYALAR = {
    "VakifBank":  os.path.join(BASE_DIR, "data", "raw", "sikayetvar_vakifbank.csv"),
    "IsBank":     os.path.join(BASE_DIR, "data", "raw", "sikayetvar_isbank.csv"),
    "KuveytTurk": os.path.join(BASE_DIR, "data", "raw", "sikayetvar_kuveyt_turk.csv"),
}
BANKA_LABEL = {
    "VakifBank":  "VakifBank",
    "IsBank":     "IsBankasi",
    "KuveytTurk": "KuveytTurk",
}
RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBank":     "#B71C1C",
    "KuveytTurk": "#1B5E20",
}

dfler = []
for key, dosya in DOSYALAR.items():
    df = pd.read_csv(dosya, encoding="utf-8")
    df["banka_key"] = key
    df["banka_label"] = BANKA_LABEL[key]
    dfler.append(df)
    print(f"  Yuklendi: {BANKA_LABEL[key]} — {len(df)} satir")

veri = pd.concat(dfler, ignore_index=True)
print(f"\nToplam: {len(veri)} satirlik birlesmis veri")

# Tarih
veri["date"] = pd.to_datetime(veri["date"], errors="coerce")
veri["ay_str"] = veri["date"].dt.strftime("%Y-%m")

# ----------------------------------------------------------
# 2. KELIME SAYISI HESABI
# ----------------------------------------------------------
# Yontem: .split() ile bosluğa gore bol, len() ile say
# Ornek: "hesabim bloke edildi" → 3 kelime
# NOT: Ham sayim. NLP on islemesi Adim 3'te yapilacak.
veri["kelime_sayisi"]   = veri["full_text"].fillna("").apply(lambda x: len(x.split()))
veri["karakter_sayisi"] = veri["full_text"].fillna("").apply(len)

# ----------------------------------------------------------
# 3. NULL DEGER RAPORU
# ----------------------------------------------------------
print("\n--- NULL DEGER RAPORU ---")
null_rapor = []
for key in ["VakifBank", "IsBank", "KuveytTurk"]:
    alt    = veri[veri["banka_key"] == key]
    toplam = len(alt)
    dolu   = alt["satisfaction"].notna().sum()
    null   = alt["satisfaction"].isna().sum()
    oran   = round(null / toplam * 100, 1)
    ortalama = round(alt["satisfaction"].mean(), 2)  # pandas skipna=True varsayilan
    null_rapor.append({
        "Banka":                       BANKA_LABEL[key],
        "Toplam Sikayet":              toplam,
        "Satisfaction Dolu":           dolu,
        "Satisfaction Null":           null,
        "Null Orani (%)":              oran,
        "Ort Satisfaction (null atlanarak)": ortalama,
    })
    print(f"  {BANKA_LABEL[key]}: {toplam} toplam | {dolu} dolu | {null} null (%{oran})")
    print(f"    Ortalama satisfaction (null atlanarak, skipna=True): {ortalama}")

null_df = pd.DataFrame(null_rapor)
null_df.to_csv("data/processed/null_raporu.csv", index=False, encoding="utf-8-sig")
print("  -> null_raporu.csv kaydedildi (utf-8-sig)")

# ----------------------------------------------------------
# 4. OZET TABLO
# ----------------------------------------------------------
ozet = veri.groupby("banka_label").agg(
    Sikayet_Sayisi      = ("id", "count"),
    Cozulme_Orani_Pct   = ("is_resolved",  lambda x: round((x == "Cozuldu").mean()*100, 2)),
    Ort_Satisfaction    = ("satisfaction", lambda x: round(x.mean(), 2)),
    Medyan_Satisfaction = ("satisfaction", lambda x: round(x.median(), 2)),
    Ort_Kelime_Sayisi   = ("kelime_sayisi","mean"),
    Ort_Goruntulenme    = ("view_count",   "mean"),
).round(2).reset_index()

ozet.to_csv("data/processed/ozet_istatistikler.csv", index=False, encoding="utf-8-sig")
print("\n--- OZET ISTATISTIKLER ---")
print(ozet.to_string(index=False))
print("  -> ozet_istatistikler.csv kaydedildi (utf-8-sig)")

# Birlesmis veriyi kaydet
veri.to_csv("data/processed/veri_ham_birlesmis.csv", index=False, encoding="utf-8-sig")
print("  -> veri_ham_birlesmis.csv kaydedildi (utf-8-sig)")

# ----------------------------------------------------------
# 5. GRAFIKLER
# ----------------------------------------------------------
banka_sirasi = ["VakifBank", "IsBank", "KuveytTurk"]
banka_labels = [BANKA_LABEL[k] for k in banka_sirasi]

# GRAFİK 1: Sikayet Sayisi
fig, ax = plt.subplots(figsize=(8, 5))
sayilar = [len(veri[veri["banka_key"] == k]) for k in banka_sirasi]
bars = ax.bar(banka_labels, sayilar,
              color=[RENKLER[k] for k in banka_sirasi],
              edgecolor="white", width=0.55)
for bar, val in zip(bars, sayilar):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 40,
            str(val), ha="center", fontsize=12, fontweight="bold")
ax.set_title("Bankaya Göre Şikayet Sayısı", fontsize=14, fontweight="bold")
ax.set_ylabel("Şikayet Sayısı")
plt.tight_layout()
plt.savefig("results/figures/01_sikayet_sayisi.png", dpi=150)
plt.close()
print("\n  Grafik 01 kaydedildi: 01_sikayet_sayisi.png")

# GRAFİK 2: Cozulme Orani
cozulme = veri.groupby(["banka_key", "is_resolved"]).size().unstack(fill_value=0)
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

# GRAFİK 3: Memnuniyet Dagilimi
fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for i, (ax, key) in enumerate(zip(axes, banka_sirasi)):
    alt    = veri[veri["banka_key"] == key]["satisfaction"].dropna()
    counts = alt.value_counts().sort_index()
    ax.bar(counts.index.astype(int), counts.values,
           color=RENKLER[key], edgecolor="white")
    ax.set_title(BANKA_LABEL[key], fontsize=12, fontweight="bold")
    ax.set_xlabel("Skor (1-5)")
    if i == 0: ax.set_ylabel("Sikayet Sayisi")
    ax.set_xticks([1, 2, 3, 4, 5])
    ort = alt.mean()
    ax.axvline(ort, color="black", linestyle="--", linewidth=1.5, label=f"Ort:{ort:.2f}")
    ax.legend(fontsize=8)
fig.suptitle("Memnuniyet Skoru Dağılımı |  NOT: Null değerler grafiğe dahil edilmedi",
             fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/03_memnuniyet_dagilimi.png", dpi=150)
plt.close()
print("  Grafik 03 kaydedildi: 03_memnuniyet_dagilimi.png")

# GRAFİK 4: Aylık şikayet trendi
aylik       = veri.groupby(["ay_str","banka_key"]).size().reset_index(name="sayi")
aylik_pivot = aylik.pivot(index="ay_str", columns="banka_key", values="sayi").fillna(0).sort_index()
fig, ax = plt.subplots(figsize=(12, 5))
for key in banka_sirasi:
    if key in aylik_pivot.columns:
        ax.plot(aylik_pivot.index, aylik_pivot[key],
                marker="o", linewidth=2, markersize=4,
                color=RENKLER[key], label=BANKA_LABEL[key])
ax.set_title("Aylık Şikayet Trendi", fontsize=14, fontweight="bold")
ax.set_xlabel("Ay"); ax.set_ylabel("Şikayet Sayısı"); ax.legend()
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("results/figures/04_aylik_trend.png", dpi=150)
plt.close()
print("  Grafik 04 kaydedildi: 04_aylik_trend.png")



# GRAFİK 6: En Sik Anahtar Kelimeler
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, (ax, key) in enumerate(zip(axes, banka_sirasi)):
    kws = veri[veri["banka_key"] == key]["keywords"].dropna()
    tum = []
    for kw in kws:
        tum.extend([k.strip().lower() for k in str(kw).split(",") if k.strip()])
    en_sik = Counter(tum).most_common(15)
    if en_sik:
        kelimeler, freqs = zip(*en_sik)
        ax.barh(list(reversed(kelimeler)), list(reversed(freqs)),
                color=RENKLER[key], edgecolor="none")
    ax.set_title(BANKA_LABEL[key], fontsize=12, fontweight="bold")
    ax.set_xlabel("Frekans")
fig.suptitle("En Sık 15 Anahtar Kelime (keywords sütunundan)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/06_anahtar_kelimeler.png", dpi=150)
plt.close()
print("Grafik 06 kaydedildi: 06_anahtar_kelimeler.png")

# GRAFİK 7: Goruntulenme Kutu Grafigi
fig, ax = plt.subplots(figsize=(9, 5))
data_bp = [veri[veri["banka_key"] == k]["view_count"].dropna().clip(upper=5000).values
           for k in banka_sirasi]
bp = ax.boxplot(data_bp, labels=banka_labels, patch_artist=True,
                medianprops=dict(color="white", linewidth=2))
for patch, key in zip(bp["boxes"], banka_sirasi):
    patch.set_facecolor(RENKLER[key])
ax.set_title("Şikayet Görüntülenme Sayısı Dağılımı (5000 üzerinde kırpıldı)",
             fontsize=12, fontweight="bold")
ax.set_ylabel("Goruntulenme Sayisi")
plt.tight_layout()
plt.savefig("results/figures/07_goruntulenme.png", dpi=150)
plt.close()
print("  Grafik 07 kaydedildi: 07_goruntulenme.png")

# ----------------------------------------------------------
# 9. ÖZET TABLO - CSV olarak kaydet
# ----------------------------------------------------------
ozet = veri.groupby("banka_key").agg(
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


print("\n" + "="*60)
print("ADIM 2 TAMAMLANDI!")
print()
print("Uretilen dosyalar:")
print("  data/processed/veri_ham_birlesmis.csv   <- sonraki adimlar bunu kullanir")
print("  data/processed/ozet_istatistikler.csv")
print("  data/processed/null_raporu.csv")
print("  results/figures/01..07_*.png")
print()
print("Excel'de Turkce gormek icin:")
print("  Dosyayi dogrudan cift tiklama ile DEGIL,")
print("  Excel > Veri > Metinden/CSV'den > ac")
print("  Adim 1'de: Dosya Kaynagi = 65001: Unicode (UTF-8) sec")
print()
print("Siradaki adim: python adim3_onisleme.py")
print("="*60)