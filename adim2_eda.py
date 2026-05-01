"""
=============================================================
ADIM 2 (İYİLEŞTİRİLMİŞ): KEŞİFSEL VERİ ANALİZİ (EDA)
=============================================================
Bankacılık Sektörü Müşteri Şikayet Analizi — 2025
Kuveyt Türk | VakıfBank | İşBankası

İYİLEŞTİRMELER:
  - Satisfaction → her yerde MEDYAN (çarpık dağılım, skew>1)
  - Mann-Whitney U testi ile istatistiksel anlamlılık
  - company_reply oranı analizi (yeni)
  - Veri kapsam farklılığı notu (aylık trend için)
  - Her analiz için ayrı, açıklayıcı Figure
  - Tüm Figure yazıları doğru Türkçe

ÜRETILEN FIGURLER:
  01 — Şikayet Sayısı (banka karşılaştırması)
  02 — Çözülme Oranı (%)
  03 — Satisfaction Dağılımı (histogram + medyan çizgisi)
  04 — Satisfaction Kutu Grafiği (çözüldü vs bilinmiyor)
  05 — Aylık Şikayet Trendi
  06 — Şirket Yanıt Oranı
  07 — Görüntülenme Dağılımı
  08 — En Sık Şikayet Kategorileri (keywords)
  09 — Veri Kalite Özet Paneli

ÇALIŞTIRMA: python adim2_eda.py
=============================================================
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 150

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
import warnings
import os

warnings.filterwarnings("ignore")

os.makedirs("results/figures", exist_ok=True)
os.makedirs("data/processed",  exist_ok=True)

# ----------------------------------------------------------
# SABITLER
# ----------------------------------------------------------
BANKA_SIRASI = ["VakifBank", "IsBank", "KuveytTurk"]
BANKA_TR = {
    "VakifBank":  "VakıfBank",
    "IsBank":     "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}
BANKA_LABELS = [BANKA_TR[b] for b in BANKA_SIRASI]

RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBank":     "#B71C1C",
    "KuveytTurk": "#1B5E20",
}

DOSYALAR = {
    "VakifBank":  "data/raw/sikayetvar_vakifbank.csv",
    "IsBank":     "data/raw/sikayetvar_isbank.csv",
    "KuveytTurk": "data/raw/sikayetvar_kuveyt_turk.csv",
}

# ----------------------------------------------------------
# 1. VERİ YÜKLEME
# ----------------------------------------------------------
print("=" * 60)
print("ADIM 2: KEŞİFSEL VERİ ANALİZİ")
print("=" * 60)
print("\n[Veri yükleniyor...]")

dfler = []
for key, dosya in DOSYALAR.items():
    df = pd.read_csv(dosya, encoding="utf-8-sig")
    df["banka_key"]   = key
    df["banka_label"] = BANKA_TR[key]
    dfler.append(df)
    print(f"  ✓ {BANKA_TR[key]:12s} → {len(df):5d} şikayet yüklendi")

veri = pd.concat(dfler, ignore_index=True)
print(f"\n  Toplam: {len(veri)} şikayet")

# Tarih dönüşümü
veri["date"]   = pd.to_datetime(veri["date"], errors="coerce")
veri["ay_str"] = veri["date"].dt.strftime("%Y-%m")

# Kelime sayısı (ham, ön işleme öncesi referans)
veri["kelime_sayisi"]   = veri["full_text"].fillna("").apply(lambda x: len(x.split()))
veri["karakter_sayisi"] = veri["full_text"].fillna("").apply(len)

# ----------------------------------------------------------
# 2. TEMELİ METRIKLER (MEDYAN POLİTİKASI)
# ----------------------------------------------------------
# Satisfaction dağılımı simetrik DEĞİL (skew > 1 → çarpık).
# Bu nedenle merkezi eğilim ölçütü olarak MEDYAN kullanılır.
# Ortalama (mean) burada yanıltıcı olur.
print("\n[Temel metrikler hesaplanıyor — Satisfaction için MEDYAN kullanılıyor...]")

for key in BANKA_SIRASI:
    alt  = veri[veri["banka_key"] == key]["satisfaction"]
    skew = alt.skew()
    print(f"  {BANKA_TR[key]:12s} | "
          f"Medyan={alt.median():.1f} | "
          f"Ortalama={alt.mean():.2f} | "
          f"Çarpıklık={skew:.2f} "
          f"{'⚠ (ortalama yanıltıcı!)' if abs(skew)>1 else ''}")

# NULL RAPORU
print("\n[NULL değer raporu...]")
null_rapor = []
for key in BANKA_SIRASI:
    alt    = veri[veri["banka_key"] == key]
    toplam = len(alt)
    sat_null  = alt["satisfaction"].isna().sum()
    kw_null   = alt["keywords"].isna().sum()
    rep_dolu  = alt["company_reply"].notna().sum()

    null_rapor.append({
        "Banka":                   BANKA_TR[key],
        "Toplam Şikayet":          toplam,
        "Satisfaction Dolu":       toplam - sat_null,
        "Satisfaction Null":       sat_null,
        "Satisfaction Null (%)":   round(sat_null / toplam * 100, 1),
        "Satisfaction Medyan":     alt["satisfaction"].median(),
        "Keywords Null":           kw_null,
        "Keywords Null (%)":       round(kw_null / toplam * 100, 1),
        "Şirket Yanıt Sayısı":     rep_dolu,
        "Şirket Yanıt Oranı (%)":  round(rep_dolu / toplam * 100, 1),
    })

null_df = pd.DataFrame(null_rapor)
null_df.to_csv("data/processed/null_raporu.csv", index=False, encoding="utf-8-sig")
print("  ✓ data/processed/null_raporu.csv kaydedildi")

# ÖZET TABLO
ozet = veri.groupby("banka_key").agg(
    Sikayet_Sayisi      = ("id",           "count"),
    Cozulme_Orani_Pct   = ("is_resolved",  lambda x: round((x == "Çözüldü").mean() * 100, 2)),
    Sat_Medyan          = ("satisfaction", "median"),   # ← MEDYAN
    Sat_Ceyrek25        = ("satisfaction", lambda x: x.quantile(0.25)),
    Sat_Ceyrek75        = ("satisfaction", lambda x: x.quantile(0.75)),
    Sat_Null_Pct        = ("satisfaction", lambda x: round(x.isna().mean() * 100, 1)),
    Ort_Kelime          = ("kelime_sayisi","mean"),
    Medyan_Goruntulenme = ("view_count",   "median"),  # ← MEDYAN (sağa çarpık)
).round(2).reset_index()

ozet.to_csv("data/processed/ozet_istatistikler.csv", index=False, encoding="utf-8-sig")
print("  ✓ data/processed/ozet_istatistikler.csv kaydedildi")

# Ham birleşik veri kaydet
veri.to_csv("data/processed/veri_ham_birlesmis.csv", index=False, encoding="utf-8-sig")
print("  ✓ data/processed/veri_ham_birlesmis.csv kaydedildi")

# ----------------------------------------------------------
# FİGÜR 01: Şikayet Sayısı
# ----------------------------------------------------------
print("\n[Figürler oluşturuluyor...]")

fig, ax = plt.subplots(figsize=(8, 5))
sayilar = [len(veri[veri["banka_key"] == k]) for k in BANKA_SIRASI]
bars = ax.bar(BANKA_LABELS, sayilar,
              color=[RENKLER[k] for k in BANKA_SIRASI],
              edgecolor="white", width=0.55)
for bar, val in zip(bars, sayilar):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 30, f"{val:,}",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_title("Bankaya Göre Şikayet Sayısı", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Şikayet Sayısı")
ax.set_ylim(0, max(sayilar) * 1.18)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/01_sikayet_sayisi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 01_sikayet_sayisi.png")

# ----------------------------------------------------------
# FİGÜR 02: Çözülme Oranı
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

coz_pct   = [round((veri[veri["banka_key"]==k]["is_resolved"]=="Çözüldü").mean()*100, 1) for k in BANKA_SIRASI]
bil_pct   = [round(100 - c, 1) for c in coz_pct]
x         = np.arange(len(BANKA_SIRASI))
genislik  = 0.35

b1 = ax.bar(x - genislik/2, coz_pct, genislik,
            label="Çözüldü", color="#43A047", edgecolor="white")
b2 = ax.bar(x + genislik/2, bil_pct, genislik,
            label="Bilinmiyor", color="#E53935", edgecolor="white")

for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.8,
            f"%{bar.get_height():.1f}", ha="center", fontsize=11, fontweight="bold", color="#43A047")
for bar in b2:
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 0.8,
            f"%{bar.get_height():.1f}", ha="center", fontsize=11, fontweight="bold", color="#E53935")

ax.set_xticks(x)
ax.set_xticklabels(BANKA_LABELS, fontsize=12)
ax.set_ylabel("Oran (%)")
ax.set_ylim(0, 100)
ax.set_title("Bankaya Göre Şikayet Çözülme Oranı (%)", fontsize=14, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=11)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/02_cozulme_orani.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 02_cozulme_orani.png")

# ----------------------------------------------------------
# FİGÜR 03: Satisfaction Dağılımı (Histogram + Medyan)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
fig.suptitle(
    "Memnuniyet Skoru Dağılımı (1–5)\n"
    "Dikey kesik çizgi: Medyan  |  Null değerler dahil edilmedi",
    fontsize=12, fontweight="bold"
)

for i, (ax, key) in enumerate(zip(axes, BANKA_SIRASI)):
    alt      = veri[veri["banka_key"] == key]["satisfaction"].dropna()
    null_pct = veri[veri["banka_key"] == key]["satisfaction"].isna().mean() * 100
    counts   = alt.value_counts().sort_index()
    medyan   = alt.median()
    skew_val = alt.skew()

    ax.bar(counts.index.astype(int), counts.values,
           color=RENKLER[key], edgecolor="white", width=0.7)
    ax.axvline(medyan, color="black", linestyle="--", linewidth=2,
               label=f"Medyan: {medyan:.0f}")

    ax.set_title(BANKA_TR[key], fontsize=12, fontweight="bold")
    ax.set_xlabel("Skor (1 = En Kötü, 5 = En İyi)")
    ax.set_ylabel("Şikayet Sayısı" if i == 0 else "")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=9)
    ax.spines[["top","right"]].set_visible(False)

    # Null oranı ve skewness notu
    ax.text(0.97, 0.97,
            f"Null: %{null_pct:.1f}\nÇarpıklık: {skew_val:.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("results/figures/03_satisfaction_dagilimi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 03_satisfaction_dagilimi.png")

# ----------------------------------------------------------
# FİGÜR 04: Satisfaction — Çözüldü vs Bilinmiyor (Kutu Grafiği + Mann-Whitney)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Çözülme Durumuna Göre Memnuniyet Skoru\n"
    "Kutu Grafiği: Medyan, Q1–Q3, Aykırı Değerler  |  ** p < 0.001 (Mann-Whitney U Testi)",
    fontsize=11, fontweight="bold"
)

for ax, key in zip(axes, BANKA_SIRASI):
    coz = veri[(veri["banka_key"]==key) & (veri["is_resolved"]=="Çözüldü")]["satisfaction"].dropna()
    bil = veri[(veri["banka_key"]==key) & (veri["is_resolved"]=="Bilinmiyor")]["satisfaction"].dropna()

    # Mann-Whitney U testi (parametrik olmayan, medyan karşılaştırması için doğru)
    mw_stat, mw_p = stats.mannwhitneyu(coz, bil, alternative="two-sided")
    p_text = f"p < 0.001" if mw_p < 0.001 else f"p = {mw_p:.3f}"

    bp = ax.boxplot([coz, bil],
                    labels=["Çözüldü", "Bilinmiyor"],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.5),
                    flierprops=dict(marker="o", markersize=2, alpha=0.3),
                    widths=0.5)

    bp["boxes"][0].set_facecolor("#43A047")
    bp["boxes"][1].set_facecolor("#E53935")

    ax.set_title(f"{BANKA_TR[key]}\n"
                 f"Çözüldü Medyan={coz.median():.0f}  |  Bilinmiyor Medyan={bil.median():.0f}",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Memnuniyet Skoru (1–5)")
    ax.set_ylim(0.5, 5.5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.spines[["top","right"]].set_visible(False)

    # İstatistiksel anlamlılık etiketi
    ax.text(1.5, 5.3, f"Mann-Whitney U\n{p_text}",
            ha="center", fontsize=8.5, color="navy",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan", alpha=0.8))

plt.tight_layout()
plt.savefig("results/figures/04_satisfaction_cozum_karsilastirma.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 04_satisfaction_cozum_karsilastirma.png")

# ----------------------------------------------------------
# FİGÜR 05: Aylık Şikayet Trendi
# ----------------------------------------------------------

aylik       = veri.groupby(["ay_str","banka_key"]).size().reset_index(name="sayi")
aylik_pivot = aylik.pivot(index="ay_str", columns="banka_key", values="sayi").sort_index()

fig, ax = plt.subplots(figsize=(13, 5))
for key in BANKA_SIRASI:
    if key in aylik_pivot.columns:
        seri  = aylik_pivot[key].dropna()
        ax.plot(seri.index, seri.values,
                marker="o", linewidth=2.2, markersize=5,
                color=RENKLER[key], label=BANKA_TR[key])
        # Son nokta etiketi
        ax.annotate(f"{int(seri.values[-1])}",
                    xy=(seri.index[-1], seri.values[-1]),
                    xytext=(5, 3), textcoords="offset points",
                    fontsize=8.5, color=RENKLER[key])

ax.set_title(
    "Aylık Şikayet Trendi\n"
    "Not: VakıfBank ve İşBankası verileri yalnızca 2025 yılının son aylarını kapsamaktadır",
    fontsize=12, fontweight="bold"
)
ax.set_xlabel("Ay")
ax.set_ylabel("Şikayet Sayısı")
ax.legend(fontsize=10)
ax.tick_params(axis="x", rotation=40, labelsize=9)
ax.spines[["top","right"]].set_visible(False)
# Kapsam farkını vurgula
ax.axvline("2025-10", color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
ax.text("2025-10", ax.get_ylim()[1]*0.92,
        "İşBankası\nveri başlangıcı", fontsize=7.5, color="darkorange", ha="center")
plt.tight_layout()
plt.savefig("results/figures/05_aylik_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 05_aylik_trend.png")

# ----------------------------------------------------------
# FİGÜR 06: Şirket Yanıt Oranı
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

yant_oran  = [veri[veri["banka_key"]==k]["company_reply"].notna().mean()*100 for k in BANKA_SIRASI]
yant_sayi  = [veri[veri["banka_key"]==k]["company_reply"].notna().sum()       for k in BANKA_SIRASI]

bars = ax.bar(BANKA_LABELS, yant_oran,
              color=[RENKLER[k] for k in BANKA_SIRASI],
              edgecolor="white", width=0.55)
for bar, oran, sayi in zip(bars, yant_oran, yant_sayi):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.2,
            f"%{oran:.1f}\n(n={sayi:,})",
            ha="center", fontsize=11, fontweight="bold")

ax.set_title("Bankaya Göre Şirket Yanıt Oranı (%)", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Yanıt Oranı (%)")
ax.set_ylim(0, 115)
ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="%50 eşiği")
ax.legend(fontsize=9)
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/06_sirket_yanit_orani.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 06_sirket_yanit_orani.png")

# ----------------------------------------------------------
# FİGÜR 07: Görüntülenme Dağılımı (Kutu Grafiği — Medyan odaklı)
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

data_bp = [veri[veri["banka_key"]==k]["view_count"].dropna().clip(upper=1000).values
           for k in BANKA_SIRASI]
bp = ax.boxplot(data_bp, labels=BANKA_LABELS, patch_artist=True,
                medianprops=dict(color="white", linewidth=2.5),
                flierprops=dict(marker="o", markersize=2, alpha=0.3),
                widths=0.5)
for patch, key in zip(bp["boxes"], BANKA_SIRASI):
    patch.set_facecolor(RENKLER[key])

medyanlar = [veri[veri["banka_key"]==k]["view_count"].median() for k in BANKA_SIRASI]
for i, (label, med) in enumerate(zip(BANKA_LABELS, medyanlar), 1):
    ax.text(i, med + 15, f"Medyan: {med:.0f}",
            ha="center", fontsize=9, color="white", fontweight="bold")

ax.set_title(
    "Şikayet Görüntülenme Sayısı Dağılımı\n(1000'in üzeri değerler 1000'e sabitlendi)",
    fontsize=12, fontweight="bold"
)
ax.set_ylabel("Görüntülenme Sayısı")
ax.spines[["top","right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/07_goruntulenme_dagilimi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 07_goruntulenme_dagilimi.png")

# ----------------------------------------------------------
# FİGÜR 08: En Sık Şikayet Kategorileri (keywords sütunundan)
# ----------------------------------------------------------

# ----------------------------------------------------------
# EŞ ANLAMLI KATEGORİ BİRLEŞTİRME HARİTALARI
# ----------------------------------------------------------
# Her banka için: { orijinal_keyword → normalleştirilmiş_kategori }
# Birleştirilmeyen kategoriler kendi adlarıyla kalır.

ESLESTIRME = {
    "VakifBank": {
        # Kredi Kartı ailesi
        "Kredi Kartı":         "Kredi Kartı",
        "Worldcard":           "Kredi Kartı",   # VakıfBank markası
        "Platinum":            "Kredi Kartı",   # kart tipi
        "Gold Kart":           "Kredi Kartı",   # kart tipi
        "Click Kart":          "Kredi Kartı",   # kart tipi
        "Troy Kart":           "Kredi Kartı",   # ödeme sistemi kartı
        "Kazandıran Tarife":   "Kredi Kartı",   # kredi kartı tarife ürünü
        "Sky Limit":           "Kredi Kartı",   # limit ürünü
        # Kredi ailesi
        "Kredi":               "Kredi",
        "İhtiyaç Kredisi":     "Kredi",
        "Bireysel Kredi":      "Kredi",
        "Konut Kredisi":       "Kredi",
        "Borç Kapatma Kredisi":"Kredi",
        "Araç Kredisi":        "Kredi",
        "Faizsiz Kredi":       "Kredi",
        # Sigorta ailesi
        "Sigorta":             "Sigorta",
        "Ferdi Kaza Sigortası":"Sigorta",
        "Sağlık Sigortası":    "Sigorta",
        "DASK":                "Sigorta",
        "Konut Sigortası":     "Sigorta",
        # Hesap ailesi
        "Hesap":               "Hesap",
        "Ek Hesap":            "Hesap",
        "Vadesiz Hesap":       "Hesap",
        "Yatırım Hesabı":      "Hesap",
        "Hesap Açma":          "Hesap",
        "Mevduat":             "Hesap",
    },
    "IsBank": {
        # Kredi Kartı ailesi (Maximum Kart = İşBankası kredi kartı markası)
        "Kredi Kartı":         "Kredi Kartı",
        "Maximum Kart":        "Kredi Kartı",   # İşBankası markası
        "Black Kart":          "Kredi Kartı",   # premium kart tipi
        "Maximum Genç Kart":   "Kredi Kartı",   # genç segmenti
        "Maximiles Kart":      "Kredi Kartı",   # mil programlı kart
        "Troy Kart":           "Kredi Kartı",
        # Hesap ailesi
        "Hesap":               "Hesap",
        "Vadesiz Hesap":       "Hesap",
        "Yatırım Hesabı":      "Hesap",
        # Kredi ailesi
        "Kredi":               "Kredi",
        "İhtiyaç Kredisi":     "Kredi",
        "Bireysel Kredi":      "Kredi",
        "Konut Kredisi":       "Kredi",
        "Anında Kredi":        "Kredi",
        "Araç Kredisi":        "Kredi",
        "Faizsiz Kredi":       "Kredi",
    },
    "KuveytTurk": {
        # Kredi Kartı ailesi
        # Sağlam Kart = KuveytTürk'ün debit/banka kartı markası
        # Banka Kartı = genel debit kart
        # Nakit Kart, Debit Kart = banka/debit kart tipleri
        # İhtiyaç Kart = kart bazlı kredi ürünü
        # Miles and Smiles = THY ortak kart
        "Kredi Kartı":         "Kredi Kartı",
        "Sağlam Kart":         "Kredi Kartı",   # KuveytTürk kart markası
        "Banka Kartı":         "Kredi Kartı",   # debit kart
        "Nakit Kart":          "Kredi Kartı",   # nakit çekim kartı
        "Debit Kart":          "Kredi Kartı",   # debit kart
        "İhtiyaç Kart":        "Kredi Kartı",   # kart ürünü
        "Miles and Smiles":    "Kredi Kartı",   # THY ortak kart
        "Troy Kart":           "Kredi Kartı",
        # Hesap ailesi
        "Hesap":               "Hesap",
        "Hesap Açma":          "Hesap",
        "Katılım Hesabı":      "Hesap",         # katılım bankacılığı hesabı
        "Birikim Hesabı":      "Hesap",
        "Yatırım Hesabı":      "Hesap",
        # Değerli Metal ailesi (KuveytTürk'e özgü güçlü kategori)
        "Altın":               "Değerli Metal",
        "Gümüş":               "Değerli Metal",
        # Kredi ailesi
        "Kredi":               "Kredi",
        "Araç Kredisi":        "Kredi",
        "Konut Kredisi":       "Kredi",
        "Faizsiz Kredi":       "Kredi",         # katılım bankası terimi
        "İhtiyaç Kredisi":     "Kredi",
        # Sigorta ailesi
        "Sigorta":             "Sigorta",
        "Konut Sigortası":     "Sigorta",
    },
}

# ----------------------------------------------------------
# VERİ YÜKLE VE NORMALİZE ET
# ----------------------------------------------------------

TOP_N = 15

# Her banka için normalize edilmiş sayımlar
normalize_sayimlar = {}
for key, dosya in DOSYALAR.items():
    df  = pd.read_csv(dosya, encoding="utf-8-sig")
    map_ = ESLESTIRME.get(key, {})
    tum  = []
    for kw in df["keywords"].dropna():
        for k in str(kw).split(","):
            k = k.strip()
            if k:
                # Eşleşme varsa normalleştirilmiş adı kullan, yoksa orijinal
                normalize = map_.get(k, k)
                tum.append(normalize)
    normalize_sayimlar[key] = Counter(tum)

# ----------------------------------------------------------
# çizim
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    "En Sık 15 Şikayet Kategorisi\n"
    "(Eş anlamlı ve banka markasına özgü terimler tek kategoride birleştirildi)",
    fontsize=13, fontweight="bold", y=1.02
)

for ax, key in zip(axes, BANKA_SIRASI):
    sayimlar = normalize_sayimlar[key]
    en_sik   = sayimlar.most_common(TOP_N)
    if not en_sik:
        continue

    kategoriler = [k for k, _ in en_sik]
    frekanslar  = [f for _, f in en_sik]
    toplam_sikayet = sum(sayimlar.values())

    # Yüzde hesapla (şikayet başına değil, keywords başına)
    y_pos = np.arange(len(kategoriler))

    # Birleştirilmiş kategorileri vurgula (renk tonu farkı)
    bar_renkleri = []
    map_ = ESLESTIRME.get(key, {})
    # Hangi kategoriler birden fazla terimden oluşuyor?
    birlesik_kategoriler = set()
    for orijinal, normalize in map_.items():
        if orijinal != normalize:
            birlesik_kategoriler.add(normalize)

    for kat in kategoriler:
        if kat in birlesik_kategoriler:
            bar_renkleri.append(RENKLER[key])        # dolu renk = birleştirilmiş
        else:
            bar_renkleri.append(RENKLER[key] + "99") # şeffaf = tek terim

    bars = ax.barh(y_pos, frekanslar,
                   color=bar_renkleri, edgecolor="none", height=0.65)

    # Değer etiketleri
    for bar, val in zip(bars, frekanslar):
        ax.text(bar.get_width() + max(frekanslar) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                str(val), va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(list(reversed(kategoriler))[::-1], fontsize=9.5)
    # Yüksek→düşük sırala (en yüksek üstte)
    ax.invert_yaxis()

    ax.set_title(BANKA_TR[key], fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Şikayet Sayısı", fontsize=10)
    ax.set_xlim(0, max(frekanslar) * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Sağ alt köşede toplam not
    ax.text(0.98, 0.01,
            f"Toplam keyword: {toplam_sikayet:,}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=7.5, color="gray")

# Açıklama kutusu (legend)
koyu_yama  = mpatches.Patch(color="#444444",       label="Birden fazla terimin birleştirildiği kategori")
acik_yama  = mpatches.Patch(color="#44444455",     label="Tek terimli kategori")
fig.legend(handles=[koyu_yama, acik_yama],
           loc="lower center", ncol=2, fontsize=9,
           bbox_to_anchor=(0.5, -0.04),
           frameon=True, edgecolor="lightgray")

plt.tight_layout()
plt.savefig("results/figures/08_sikayet_kategorileri.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("✓ 08_sikayet_kategorileri.png kaydedildi")

# Konsol özet
print()
for key in BANKA_SIRASI:
    print(f"=== {BANKA_TR[key]} — Birleştirilen kategoriler ===")
    map_ = ESLESTIRME.get(key, {})
    from collections import defaultdict
    ters = defaultdict(list)
    for orijinal, normalize in map_.items():
        if orijinal != normalize:
            ters[normalize].append(orijinal)
    for hedef, kaynaklar in sorted(ters.items()):
        print(f"  '{hedef}' ← {', '.join(kaynaklar)}")
    print()
"""
=== VakıfBank — Birleştirilen kategoriler ===
  'Hesap' ← Ek Hesap, Vadesiz Hesap, Yatırım Hesabı, Hesap Açma, Mevduat
  'Kredi' ← İhtiyaç Kredisi, Bireysel Kredi, Konut Kredisi, Borç Kapatma Kredisi, Araç Kredisi, Faizsiz Kredi
  'Kredi Kartı' ← Worldcard, Platinum, Gold Kart, Click Kart, Troy Kart, Kazandıran Tarife, Sky Limit
  'Sigorta' ← Ferdi Kaza Sigortası, Sağlık Sigortası, DASK, Konut Sigortası

=== İşBankası — Birleştirilen kategoriler ===
  'Hesap' ← Vadesiz Hesap, Yatırım Hesabı
  'Kredi' ← İhtiyaç Kredisi, Bireysel Kredi, Konut Kredisi, Anında Kredi, Araç Kredisi, Faizsiz Kredi
  'Kredi Kartı' ← Maximum Kart, Black Kart, Maximum Genç Kart, Maximiles Kart, Troy Kart

=== Kuveyt Türk — Birleştirilen kategoriler ===
  'Değerli Metal' ← Altın, Gümüş
  'Hesap' ← Hesap Açma, Katılım Hesabı, Birikim Hesabı, Yatırım Hesabı
  'Kredi' ← Araç Kredisi, Konut Kredisi, Faizsiz Kredi, İhtiyaç Kredisi
  'Kredi Kartı' ← Sağlam Kart, Banka Kartı, Nakit Kart, Debit Kart, İhtiyaç Kart, Miles and Smiles, Troy Kart
  'Sigorta' ← Konut Sigortası

"""
# ----------------------------------------------------------
# FİGÜR 09: Veri Kalite Özet Paneli
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Veri Kalite Özet Paneli", fontsize=15, fontweight="bold", y=1.01)

for col_i, key in enumerate(BANKA_SIRASI):
    alt = veri[veri["banka_key"] == key]
    toplam = len(alt)

    # Üst satır: Null oranları
    ax = axes[0, col_i]
    sutunlar   = ["satisfaction", "keywords", "company_reply", "view_count"]
    etiketler  = ["Memnuniyet\nSkoru", "Şikayet\nKategorisi", "Şirket\nYanıtı", "Görüntülenme\nSayısı"]
    null_pct   = []
    for s in sutunlar:
        if s == "company_reply":
            # company_reply için null = yanıt yok
            null_pct.append(alt[s].isna().mean() * 100)
        else:
            null_pct.append(alt[s].isna().mean() * 100)

    renkler_bar = ["#E53935" if p > 20 else "#FFA000" if p > 5 else "#43A047"
                   for p in null_pct]
    bars = ax.bar(etiketler, null_pct, color=renkler_bar, edgecolor="white", width=0.6)
    for bar, val in zip(bars, null_pct):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5, f"%{val:.1f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_title(f"{BANKA_TR[key]}\nNull/Eksik Değer Oranları",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Eksik Oran (%)")
    ax.set_ylim(0, max(null_pct) * 1.3 + 5)
    ax.spines[["top","right"]].set_visible(False)

    # Alt satır: Sınıf dengesi
    ax2 = axes[1, col_i]
    coz_n = (alt["is_resolved"] == "Çözüldü").sum()
    bil_n = toplam - coz_n
    wedges, texts, autotexts = ax2.pie(
        [coz_n, bil_n],
        labels=["Çözüldü", "Bilinmiyor"],
        colors=["#43A047", "#E53935"],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for autotext in autotexts:
        autotext.set_fontsize(10)
        autotext.set_fontweight("bold")
    ax2.set_title(f"{BANKA_TR[key]}\nÇözülme Durumu Dağılımı\n(n={toplam:,})",
                  fontsize=10, fontweight="bold")

# Açıklama notu
fig.text(0.5, -0.02,
         "Null Oran Renk Kodu:  🟢 <%5 (İyi)  🟡 %5–20 (Dikkat)  🔴 >%20 (Yüksek)",
         ha="center", fontsize=10, color="gray")

plt.tight_layout()
plt.savefig("results/figures/09_veri_kalite_panel.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 09_veri_kalite_panel.png")

# ----------------------------------------------------------
# TERMİNAL ÖZET
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("ÖZET İSTATİSTİKLER (Satisfaction = MEDYAN)")
print("=" * 60)
print(ozet[["banka_key","Sikayet_Sayisi","Cozulme_Orani_Pct",
            "Sat_Medyan","Sat_Null_Pct","Medyan_Goruntulenme"]].to_string(index=False))

print("\n" + "=" * 60)
print("ADIM 2 TAMAMLANDI!")
print("=" * 60)
print("""
Üretilen dosyalar:
  data/processed/veri_ham_birlesmis.csv   ← Adım 3 bunu kullanır
  data/processed/ozet_istatistikler.csv
  data/processed/null_raporu.csv
  results/figures/01_sikayet_sayisi.png
  results/figures/02_cozulme_orani.png
  results/figures/03_satisfaction_dagilimi.png
  results/figures/04_satisfaction_cozum_karsilastirma.png
  results/figures/05_aylik_trend.png
  results/figures/06_sirket_yanit_orani.png
  results/figures/07_goruntulenme_dagilimi.png
  results/figures/08_sikayet_kategorileri.png
  results/figures/09_veri_kalite_panel.png

Sıradaki adım: python adim3_onisleme.py
""")
