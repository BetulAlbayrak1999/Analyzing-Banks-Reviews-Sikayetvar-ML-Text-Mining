"""
=============================================================
ADIM 2: KEŞİFSEL VERİ ANALİZİ (EDA)
=============================================================
Bankacılık Sektörü Müşteri Şikayet Analizi — 2026
Kuveyt Türk | VakıfBank | İşBankası

TEMEL KARARLAR:
  - view_count string olarak gelebiliyor → pd.to_numeric ile zorla
  - Satisfaction dağılımı çarpık (skew > 1) → MEDYAN kullanılır
  - is_resolved: 2 sınıf → Çözüldü / Çözülmedi
  - Mann-Whitney U testi (2 grup, parametrik olmayan)
  - Keywords: # prefix temizleme + eş anlamlı kategori birleştirme
  - Tam yıl veri (Ocak–Aralık 2026), aylık trend yorumu önemli

ÜRETILEN FIGURLER:
  01 — Şikayet Sayısı
  02 — Çözülme Durumu (yüzde + mutlak sayı)
  03 — Memnuniyet Skoru Dağılımı
  04 — Memnuniyet & Çözülme Durumu (Mann-Whitney)
  05 — Aylık Şikayet Trendi
  06 — Şirket Yanıt Oranı
  07 — Görüntülenme Sayısı Dağılımı
  08 — En Sık Şikayet Kategorileri (eş anlamlılar birleştirilmiş)
  09 — Veri Kalite Özet Paneli

ÇALIŞTIRMA: python adim2_eda.py
=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 150

import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter, defaultdict
import warnings, os

warnings.filterwarnings("ignore")
os.makedirs("results/figures", exist_ok=True)
os.makedirs("data/processed",  exist_ok=True)

# ===========================================================
# SABİTLER
# ===========================================================
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
    "VakifBank":  "data/raw/vakifbank_2026.csv",
    "IsBank":     "data/raw/is-bankasi_2026.csv",
    "KuveytTurk": "data/raw/kuveyt-turk_2026.csv",
}
# is_resolved: 2 sınıf (yeni veride Bilinmiyor yok)
COZUM_RENK = {
    "Çözüldü":  "#43A047",
    "Çözülmedi": "#E53935",
}

# ===========================================================
# EŞ ANLAMLI KATEGORİ BİRLEŞTİRME HARİTALARI
# ----------------------------------------------------------
# Her banka için: orijinal_keyword → normalleştirilmiş_kategori
#
# Adım 1: # prefix kaldır  ('#Kredi Kartı' → 'Kredi Kartı')
# Adım 2: Bu harita ile eş anlamlı/marka bazlı kategorileri birleştir
#
# VakıfBank Kredi Kartı ailesi:
#   Worldcard, Platinum, Gold Kart, Click Kart, Troy Kart,
#   Kazandıran Tarife (kart tarife ürünü), Sky Limit (limit ürünü),
#   Banka Kartı (debit kart)
#
# İşBankası Kredi Kartı ailesi:
#   Maximum Kart (İşBankası'nın kendi kart markası),
#   Black Kart (premium), Maximum Genç Kart, Banka Kartı,
#   Sanal Kart, Troy Kart
#
# Kuveyt Türk Kredi Kartı ailesi:
#   Sağlam Kart (KuveytTürk'ün kart markası),
#   Banka Kartı (debit), İhtiyaç Kart, Nakit Kart,
#   Debit Kart, Miles and Smiles (THY ortak kart), Troy Kart
#
# Kuveyt Türk Değerli Metal:
#   Altın + Gümüş → KuveytTürk'e özgü güçlü kategori
#   (katılım bankası olarak altın işlemleri önemli)
#
# Kuveyt Türk Döviz:
#   Dolar + Euro + Döviz → Döviz İşlemleri
# ===========================================================
ESLESTIRME = {
    "VakifBank": {
        # Kredi Kartı ailesi
        "Kredi Kartı":           "Kredi Kartı",
        "Worldcard":             "Kredi Kartı",
        "Platinum":              "Kredi Kartı",
        "Gold Kart":             "Kredi Kartı",
        "Click Kart":            "Kredi Kartı",
        "Troy Kart":             "Kredi Kartı",
        "Kazandıran Tarife":     "Kredi Kartı",
        "Sky Limit":             "Kredi Kartı",
        "Banka Kartı":           "Kredi Kartı",   # debit kart
        # Kredi ailesi
        "Kredi":                 "Kredi",
        "İhtiyaç Kredisi":       "Kredi",
        "Bireysel Kredi":        "Kredi",
        "Konut Kredisi":         "Kredi",
        "Borç Kapatma Kredisi":  "Kredi",
        "Araç Kredisi":          "Kredi",
        "Faizsiz Kredi":         "Kredi",
        # Sigorta ailesi
        "Sigorta":               "Sigorta",
        "Ferdi Kaza Sigortası":  "Sigorta",
        "Sağlık Sigortası":      "Sigorta",
        "DASK":                  "Sigorta",
        "Konut Sigortası":       "Sigorta",
        # Hesap ailesi
        "Hesap":                 "Hesap",
        "Ek Hesap":              "Hesap",
        "Vadesiz Hesap":         "Hesap",
        "Yatırım Hesabı":        "Hesap",
        "Hesap Açma":            "Hesap",
        "Mevduat":               "Hesap",
    },
    "IsBank": {
        # Kredi Kartı ailesi
        # Maximum Kart = İşBankası kredi kartı markası
        "Kredi Kartı":           "Kredi Kartı",
        "Maximum Kart":          "Kredi Kartı",
        "Black Kart":            "Kredi Kartı",
        "Maximum Genç Kart":     "Kredi Kartı",
        "Maximiles Kart":        "Kredi Kartı",
        "Banka Kartı":           "Kredi Kartı",
        "Sanal Kart":            "Kredi Kartı",
        "Troy Kart":             "Kredi Kartı",
        "Gold Kart":             "Kredi Kartı",
        # Hesap ailesi
        "Hesap":                 "Hesap",
        "Vadesiz Hesap":         "Hesap",
        "Yatırım Hesabı":        "Hesap",
        # Kredi ailesi
        "Kredi":                 "Kredi",
        "İhtiyaç Kredisi":       "Kredi",
        "Bireysel Kredi":        "Kredi",
        "Konut Kredisi":         "Kredi",
        "Anında Kredi":          "Kredi",
        "Araç Kredisi":          "Kredi",
        "Faizsiz Kredi":         "Kredi",
        # Sigorta/Emeklilik
        "Sigorta":               "Sigorta",
        "Emeklilik":             "Sigorta",
    },
    "KuveytTurk": {
        # Kredi Kartı ailesi
        # Sağlam Kart = KuveytTürk kart markası
        "Kredi Kartı":           "Kredi Kartı",
        "Sağlam Kart":           "Kredi Kartı",
        "Banka Kartı":           "Kredi Kartı",
        "Nakit Kart":            "Kredi Kartı",
        "Debit Kart":            "Kredi Kartı",
        "İhtiyaç Kart":          "Kredi Kartı",
        "Miles and Smiles":      "Kredi Kartı",   # THY ortak kart
        "Troy Kart":             "Kredi Kartı",
        # Hesap ailesi
        "Hesap":                 "Hesap",
        "Hesap Açma":            "Hesap",
        "Katılım Hesabı":        "Hesap",         # katılım bankasına özgü
        "Birikim Hesabı":        "Hesap",
        "Yatırım Hesabı":        "Hesap",
        # Değerli Metal ailesi (KuveytTürk'e özgü — katılım bankası)
        "Altın":                 "Değerli Metal",
        "Gümüş":                 "Değerli Metal",
        # Döviz ailesi
        "Dolar":                 "Döviz İşlemleri",
        "Euro":                  "Döviz İşlemleri",
        "Döviz":                 "Döviz İşlemleri",
        # Kredi ailesi
        "Kredi":                 "Kredi",
        "Araç Kredisi":          "Kredi",
        "Konut Kredisi":         "Kredi",
        "Faizsiz Kredi":         "Kredi",         # katılım bankası terimi
        "İhtiyaç Kredisi":       "Kredi",
        # Sigorta/Emeklilik ailesi
        "Sigorta":               "Sigorta",
        "Konut Sigortası":       "Sigorta",
        "Bireysel Emeklilik":    "Sigorta",
    },
}


# ===========================================================
# YARDIMCI FONKSİYON
# ===========================================================
def keyword_normalize(kw_serisi, banka_key):
    """
    keywords sütununu normalize eder:
      1. # prefix kaldır  ('#Kredi Kartı' → 'Kredi Kartı')
      2. Eş anlamlı/marka kategorileri birleştir (ESLESTIRME haritası)
    """
    map_ = ESLESTIRME.get(banka_key, {})
    tum  = []
    for kw in kw_serisi.dropna():
        for k in str(kw).split(","):
            k = k.strip().lstrip("#").strip()
            if k:
                tum.append(map_.get(k, k))
    return Counter(tum)


# ===========================================================
# 1. VERİ YÜKLEME
# ===========================================================
print("=" * 65)
print("ADIM 2: KEŞİFSEL VERİ ANALİZİ (EDA)")
print("=" * 65)
print("\n[Veri yükleniyor...]")

dfler = []
for key, dosya in DOSYALAR.items():
    df = pd.read_csv(dosya, encoding="utf-8-sig")
    df["banka_key"]   = key
    df["banka_label"] = BANKA_TR[key]
    dfler.append(df)
    print(f"  ✓ {BANKA_TR[key]:12s} → {len(df):5,d} şikayet")

veri = pd.concat(dfler, ignore_index=True)
print(f"\n  Toplam: {len(veri):,} şikayet")

# ===========================================================
# 2. SAYISAL SÜTUN DÖNÜŞÜMÜ
# view_count bazı CSV'lerde string olarak geliyor ('3383').
# Tüm sayısal sütunlar güvenle pd.to_numeric ile dönüştürülür.
# ===========================================================
print("\n[Sayısal sütunlar dönüştürülüyor...]")
for sutun in ["satisfaction", "view_count", "upvote_count"]:
    if sutun not in veri.columns:
        continue
    # str.strip() → baştaki/sondaki boşlukları temizle
    # str.replace(',', '.') → ondalık virgül varsa nokta yap
    # replace 'nan'/''/None → np.nan
    veri[sutun] = (
        veri[sutun].astype(str).str.strip()
        .str.replace(",", ".", regex=False)
        .replace({"nan": np.nan, "": np.nan, "None": np.nan})
    )
    veri[sutun] = pd.to_numeric(veri[sutun], errors="coerce")
    print(f"  ✓ {sutun:15s} → numeric  "
          f"(null: {veri[sutun].isna().sum():,}, "
          f"medyan: {veri[sutun].median():.0f})")

# ===========================================================
# 3. TEMEL DÖNÜŞÜMLER
# ===========================================================
veri["date"]          = pd.to_datetime(veri["date"], errors="coerce")
veri["ay_str"]        = veri["date"].dt.strftime("%Y-%m")
veri["kelime_sayisi"] = veri["full_text"].fillna("").apply(
    lambda x: len(x.split()))

print(f"\n  is_resolved dağılımı:")
print(veri["is_resolved"].value_counts().to_string())

# ===========================================================
# 4. ÇARPIKLIK KONTROLÜ — MEDYAN KULLANIM GEREKÇESİ
# ===========================================================
print("\n[Satisfaction çarpıklık analizi — MEDYAN kullanım gerekçesi...]")
for key in BANKA_SIRASI:
    s    = veri[veri["banka_key"] == key]["satisfaction"].dropna()
    skew = s.skew()
    print(f"  {BANKA_TR[key]:12s} | Medyan={s.median():.1f} | "
          f"Ortalama={s.mean():.2f} | Çarpıklık={skew:.2f} "
          f"{'⚠ Ortalama yanıltıcı!' if abs(skew) > 1 else ''}")

# ===========================================================
# 5. RAPORLAR
# ===========================================================
print("\n[Raporlar kaydediliyor...]")

null_satirlar, ozet_satirlar = [], []
for key in BANKA_SIRASI:
    alt = veri[veri["banka_key"] == key]
    n   = len(alt)
    coz = (alt["is_resolved"] == "Çözüldü").sum()
    coz_deg = (alt["is_resolved"] == "Çözülmedi").sum()

    null_satirlar.append({
        "Banka":                  BANKA_TR[key],
        "Toplam Şikayet":         n,
        "Satisfaction Null":      int(alt["satisfaction"].isna().sum()),
        "Satisfaction Null (%)":  round(alt["satisfaction"].isna().mean()*100, 1),
        "Satisfaction Medyan":    alt["satisfaction"].median(),
        "Keywords Null":          int(alt["keywords"].isna().sum()),
        "Keywords Null (%)":      round(alt["keywords"].isna().mean()*100, 1),
        "Şirket Yanıt Sayısı":    int(alt["company_reply"].notna().sum()),
        "Şirket Yanıt Oranı (%)": round(alt["company_reply"].notna().mean()*100, 1),
    })
    ozet_satirlar.append({
        "Banka":               BANKA_TR[key],
        "Şikayet Sayısı":      n,
        "Çözüldü (%)":         round(coz / n * 100, 2),
        "Çözülmedi (%)":       round(coz_deg / n * 100, 2),
        "Satisfaction Medyan": alt["satisfaction"].median(),
        "Satisfaction Q1":     alt["satisfaction"].quantile(0.25),
        "Satisfaction Q3":     alt["satisfaction"].quantile(0.75),
        "Görüntülenme Medyan": alt["view_count"].median(),
        "Ort Kelime Sayısı":   round(alt["kelime_sayisi"].mean(), 1),
    })

pd.DataFrame(null_satirlar).to_csv(
    "data/processed/null_raporu.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(ozet_satirlar).to_csv(
    "data/processed/ozet_istatistikler.csv", index=False, encoding="utf-8-sig")
veri.to_csv(
    "data/processed/veri_ham_birlesmis.csv", index=False, encoding="utf-8-sig")
print("  ✓ null_raporu.csv | ozet_istatistikler.csv | veri_ham_birlesmis.csv")

ozet = pd.DataFrame(ozet_satirlar)

# ===========================================================
# FİGÜRLER
# ===========================================================
print("\n[Figürler oluşturuluyor...]")

# ----------------------------------------------------------
# FİGÜR 01: Şikayet Sayısı
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
sayilar = [len(veri[veri["banka_key"] == k]) for k in BANKA_SIRASI]
bars = ax.bar(BANKA_LABELS, sayilar,
              color=[RENKLER[k] for k in BANKA_SIRASI],
              edgecolor="white", width=0.55)
for bar, val in zip(bars, sayilar):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 30, f"{val:,}",
            ha="center", va="bottom", fontsize=13, fontweight="bold")
ax.set_title("Bankaya Göre Toplam Şikayet Sayısı", fontsize=14,
             fontweight="bold", pad=12)
ax.set_ylabel("Şikayet Sayısı")
ax.set_ylim(0, max(sayilar) * 1.18)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/01_sikayet_sayisi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 01_sikayet_sayisi.png")

# ----------------------------------------------------------
# FİGÜR 02: Çözülme Durumu (Yüzde + Mutlak Sayı)
# ----------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Çözülme Durumu Analizi — Çözüldü / Çözülmedi",
             fontsize=13, fontweight="bold")

x = np.arange(len(BANKA_SIRASI))

# Sol: Yüzde yığılmış çubuk
alt_ = np.zeros(len(BANKA_SIRASI))
for durum in ["Çözüldü", "Çözülmedi"]:
    vals = [(veri[veri["banka_key"]==k]["is_resolved"]==durum).mean()*100
            for k in BANKA_SIRASI]
    bars1 = ax1.bar(x, vals, bottom=alt_, label=durum,
                    color=COZUM_RENK[durum], edgecolor="white", width=0.55)
    for xi, (val, bot) in enumerate(zip(vals, alt_)):
        if val > 3:
            ax1.text(xi, bot + val/2, f"%{val:.1f}",
                     ha="center", va="center", fontsize=11,
                     fontweight="bold", color="white")
    alt_ = [a + v for a, v in zip(alt_, vals)]

ax1.set_xticks(x)
ax1.set_xticklabels(BANKA_LABELS, fontsize=12)
ax1.set_ylabel("Oran (%)")
ax1.set_ylim(0, 108)
ax1.set_title("Oran Dağılımı (%)", fontsize=11, fontweight="bold")
ax1.legend(loc="lower right", fontsize=10)
ax1.spines[["top", "right"]].set_visible(False)

# Sağ: Mutlak sayı gruplanmış çubuk
genislik = 0.32
ofset    = [-genislik/2, genislik/2]
for di, durum in enumerate(["Çözüldü", "Çözülmedi"]):
    vals = [(veri[veri["banka_key"]==k]["is_resolved"]==durum).sum()
            for k in BANKA_SIRASI]
    bars2 = ax2.bar(x + ofset[di], vals, genislik,
                    label=durum, color=COZUM_RENK[durum], edgecolor="white")
    for bar, val in zip(bars2, vals):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 15, f"{val:,}",
                 ha="center", fontsize=9, fontweight="bold")
ax2.set_xticks(x)
ax2.set_xticklabels(BANKA_LABELS, fontsize=12)
ax2.set_ylabel("Şikayet Sayısı")
ax2.set_title("Mutlak Sayı", fontsize=11, fontweight="bold")
ax2.legend(loc="upper right", fontsize=10)
ax2.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.savefig("results/figures/02_cozulme_durumu.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 02_cozulme_durumu.png")

# ----------------------------------------------------------
# FİGÜR 03: Memnuniyet Skoru Dağılımı (Histogram + Medyan)
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle(
    "Memnuniyet Skoru Dağılımı (1–5)\n"
    "Kesik çizgi: Medyan  |  Null değerler dahil edilmedi",
    fontsize=12, fontweight="bold"
)
for i, (ax, key) in enumerate(zip(axes, BANKA_SIRASI)):
    alt      = veri[veri["banka_key"] == key]["satisfaction"].dropna()
    null_pct = veri[veri["banka_key"] == key]["satisfaction"].isna().mean() * 100
    counts   = alt.value_counts().sort_index()
    medyan   = alt.median()

    ax.bar(counts.index.astype(int), counts.values,
           color=RENKLER[key], edgecolor="white", width=0.7)
    ax.axvline(medyan, color="black", linestyle="--",
               linewidth=2, label=f"Medyan: {medyan:.0f}")
    ax.set_title(BANKA_TR[key], fontsize=12, fontweight="bold")
    ax.set_xlabel("Skor (1 = En Kötü, 5 = En İyi)")
    ax.set_ylabel("Şikayet Sayısı" if i == 0 else "")
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.97, 0.97,
            f"Null: %{null_pct:.1f}\nÇarpıklık: {alt.skew():.2f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, color="gray",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig("results/figures/03_satisfaction_dagilimi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 03_satisfaction_dagilimi.png")

# ----------------------------------------------------------
# FİGÜR 04: Memnuniyet — Çözüldü vs Çözülmedi
# Mann-Whitney U testi (2 grup, parametrik olmayan — doğru seçim)
# ----------------------------------------------------------
"""fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.suptitle(
    "Çözülme Durumuna Göre Memnuniyet Skoru\n"
    "Mann-Whitney U Testi (parametrik olmayan, 2 grup karşılaştırması)",
    fontsize=11, fontweight="bold"
)
for ax, key in zip(axes, BANKA_SIRASI):
    alt = veri[veri["banka_key"] == key]
    coz  = alt[alt["is_resolved"] == "Çözüldü"]["satisfaction"].dropna()
    cdeg = alt[alt["is_resolved"] == "Çözülmedi"]["satisfaction"].dropna()

    if len(coz) < 5 or len(cdeg) < 5:
        ax.text(0.5, 0.5, "Yeterli veri yok",
                ha="center", transform=ax.transAxes)
        continue

    mw_stat, mw_p = stats.mannwhitneyu(coz, cdeg, alternative="two-sided")
    p_text = "p < 0.001" if mw_p < 0.001 else f"p = {mw_p:.3f}"

    bp = ax.boxplot(
        [coz, cdeg],
        labels=["Çözüldü", "Çözülmedi"],
        patch_artist=True,
        medianprops=dict(color="white", linewidth=2.5),
        flierprops=dict(marker="o", markersize=2, alpha=0.3),
        widths=0.5
    )
    bp["boxes"][0].set_facecolor(COZUM_RENK["Çözüldü"])
    bp["boxes"][1].set_facecolor(COZUM_RENK["Çözülmedi"])

    ax.set_title(
        f"{BANKA_TR[key]}\n"
        f"Çözüldü Med={coz.median():.0f}  |  Çözülmedi Med={cdeg.median():.0f}",
        fontsize=10, fontweight="bold"
    )
    ax.set_ylabel("Memnuniyet Skoru (1–5)")
    ax.set_ylim(0.5, 6.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.5, 0.97, f"Mann-Whitney U\n{p_text}",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=9, color="navy",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="lightcyan", alpha=0.85))

plt.tight_layout()
plt.savefig("results/figures/04_satisfaction_cozum_karsilastirma.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 04_satisfaction_cozum_karsilastirma.png")
"""
# ----------------------------------------------------------
# FİGÜR 05: Aylık Şikayet Trendi
# ----------------------------------------------------------
aylik       = veri.groupby(["ay_str","banka_key"]).size().reset_index(name="sayi")
aylik_pivot = aylik.pivot(index="ay_str", columns="banka_key",
                          values="sayi").sort_index()

AY_TR = {
    "2026-01": "Oca", "2026-02": "Şub", "2026-03": "Mar",
    "2026-04": "Nis", "2026-05": "May", "2026-06": "Haz",
    "2026-07": "Tem", "2026-08": "Ağu", "2026-09": "Eyl",
    "2026-10": "Eki", "2026-11": "Kas", "2026-12": "Ara",
}
x_etiketler = [AY_TR.get(a, a) for a in aylik_pivot.index]

fig, ax = plt.subplots(figsize=(13, 5))
x_pos = np.arange(len(aylik_pivot.index))
for key in BANKA_SIRASI:
    if key not in aylik_pivot.columns:
        continue
    seri = aylik_pivot[key].fillna(0)
    ax.plot(x_pos, seri.values,
            marker="o", linewidth=2.2, markersize=5,
            color=RENKLER[key], label=BANKA_TR[key])
    # Son değer etiketi
    ax.annotate(f"{int(seri.values[-1]):,}",
                xy=(x_pos[-1], seri.values[-1]),
                xytext=(4, 4), textcoords="offset points",
                fontsize=8.5, color=RENKLER[key], fontweight="bold")

ax.set_xticks(x_pos)
ax.set_xticklabels(x_etiketler, fontsize=10)
ax.set_title("Aylık Şikayet Trendi (Ocak–Nisan 2026)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Ay (2026)")
ax.set_ylabel("Şikayet Sayısı")
ax.legend(fontsize=10)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/05_aylik_trend.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 05_aylik_trend.png")

# ----------------------------------------------------------
# FİGÜR 06: Şirket Yanıt Oranı
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
yant_pct  = [veri[veri["banka_key"]==k]["company_reply"].notna().mean()*100
             for k in BANKA_SIRASI]
yant_sayi = [veri[veri["banka_key"]==k]["company_reply"].notna().sum()
             for k in BANKA_SIRASI]

bars = ax.bar(BANKA_LABELS, yant_pct,
              color=[RENKLER[k] for k in BANKA_SIRASI],
              edgecolor="white", width=0.55)
for bar, pct, sayi in zip(bars, yant_pct, yant_sayi):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + 1.5,
            f"%{pct:.1f}\n(n={sayi:,})",
            ha="center", fontsize=11, fontweight="bold")

ax.set_title("Bankaya Göre Şirket Yanıt Oranı (%)",
             fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Yanıt Oranı (%)")
ax.set_ylim(0, 115)
ax.axhline(50, color="gray", linestyle="--",
           linewidth=1, alpha=0.5, label="%50 eşiği")
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/06_sirket_yanit_orani.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 06_sirket_yanit_orani.png")

# ----------------------------------------------------------
# FİGÜR 07: Görüntülenme Sayısı Dağılımı
# ----------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))

# %95 yüzdelik dilim ile sınırla (aykırı değer baskısı)
clip_val = int(veri["view_count"].quantile(0.95))
data_bp  = [veri[veri["banka_key"]==k]["view_count"]
            .dropna().clip(upper=clip_val).values
            for k in BANKA_SIRASI]
bp = ax.boxplot(data_bp, labels=BANKA_LABELS, patch_artist=True,
                medianprops=dict(color="white", linewidth=2.5),
                flierprops=dict(marker="o", markersize=2, alpha=0.3),
                widths=0.5)
for patch, key in zip(bp["boxes"], BANKA_SIRASI):
    patch.set_facecolor(RENKLER[key])

# Medyan etiketleri
for i, key in enumerate(BANKA_SIRASI, 1):
    med = veri[veri["banka_key"]==key]["view_count"].median()
    ax.text(i, med + clip_val*0.015,
            f"Medyan:\n{med:,.0f}",
            ha="center", fontsize=8, color="white",
            fontweight="bold", va="bottom")

ax.set_title(
    f"Şikayet Görüntülenme Sayısı Dağılımı\n"
    f"(Değerler {clip_val:,} ile sınırlandırıldı — %95 yüzdelik dilim)",
    fontsize=12, fontweight="bold"
)
ax.set_ylabel("Görüntülenme Sayısı")
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("results/figures/07_goruntulenme_dagilimi.png", dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 07_goruntulenme_dagilimi.png")

# ----------------------------------------------------------
# FİGÜR 08: En Sık Şikayet Kategorileri
# Eş anlamlı + banka markasına özgü terimler birleştirildi
# ----------------------------------------------------------
TOP_N = 15

normalize_sayimlar = {}
birlesik_kategoriler = {}
for key in BANKA_SIRASI:
    alt = veri[veri["banka_key"] == key]
    normalize_sayimlar[key] = keyword_normalize(alt["keywords"], key)
    # Hangi hedef kategoriler birden fazla kaynaktan besleniyor?
    map_  = ESLESTIRME.get(key, {})
    ters_ = defaultdict(list)
    for orijinal, hedef in map_.items():
        if orijinal != hedef:
            ters_[hedef].append(orijinal)
    birlesik_kategoriler[key] = set(ters_.keys())

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
fig.suptitle(
    "En Sık 15 Şikayet Kategorisi\n"
    "Koyu renk: Birden fazla terimin birleştirildiği kategori  "
    "|  Açık renk: Tek terimli kategori",
    fontsize=12, fontweight="bold", y=1.02
)

for ax, key in zip(axes, BANKA_SIRASI):
    sayimlar = normalize_sayimlar[key]
    en_sik   = sayimlar.most_common(TOP_N)
    if not en_sik:
        continue

    kategoriler = [k for k, _ in en_sik]
    frekanslar  = [f for _, f in en_sik]
    maks        = max(frekanslar)

    bar_renkleri = [
        RENKLER[key] if kat in birlesik_kategoriler[key]
        else RENKLER[key] + "70"
        for kat in kategoriler
    ]

    y_pos = np.arange(len(kategoriler))
    ax.barh(y_pos, frekanslar, color=bar_renkleri,
            edgecolor="none", height=0.65)
    for yi, (val, kat) in enumerate(zip(frekanslar, kategoriler)):
        ax.text(val + maks*0.012, yi, str(val),
                va="center", fontsize=8.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(kategoriler, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_title(BANKA_TR[key], fontsize=13, fontweight="bold", pad=8)
    ax.set_xlabel("Şikayet Sayısı", fontsize=10)
    ax.set_xlim(0, maks * 1.22)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    # Birleştirilen kategori sayısı notu
    n_birl = len(birlesik_kategoriler[key])
    ax.text(0.98, 0.01, f"{n_birl} kategori birleştirildi",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="gray")

# Ortak legend
koyu  = mpatches.Patch(color="#555555",
                        label="Birden fazla terimin birleştirildiği kategori")
acik  = mpatches.Patch(color="#55555555",
                        label="Tek terimli kategori")
fig.legend(handles=[koyu, acik], loc="lower center", ncol=2,
           fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=True)

plt.tight_layout()
plt.savefig("results/figures/08_sikayet_kategorileri.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 08_sikayet_kategorileri.png")

# ----------------------------------------------------------
# FİGÜR 09: Veri Kalite Özet Paneli
# ----------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Veri Kalite Özet Paneli", fontsize=15, fontweight="bold", y=1.01)

for col_i, key in enumerate(BANKA_SIRASI):
    alt    = veri[veri["banka_key"] == key]
    toplam = len(alt)

    # Üst: Eksik değer oranları
    ax_ust = axes[0, col_i]
    sutunlar  = ["satisfaction", "keywords", "company_reply", "view_count"]
    etiketler = ["Memnuniyet\nSkoru", "Şikayet\nKategorisi",
                 "Şirket\nYanıtı*", "Görüntülenme\nSayısı"]
    null_pct = [alt[s].isna().mean() * 100 for s in sutunlar]
    renkler_bar = [
        "#E53935" if p > 20 else "#FFA000" if p > 5 else "#43A047"
        for p in null_pct
    ]
    bars_k = ax_ust.bar(etiketler, null_pct, color=renkler_bar,
                        edgecolor="white", width=0.6)
    for bar, val in zip(bars_k, null_pct):
        ax_ust.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.5,
                    f"%{val:.1f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax_ust.set_title(f"{BANKA_TR[key]}\nEksik Değer Oranları (%)",
                     fontsize=10, fontweight="bold")
    ax_ust.set_ylabel("Eksik Oran (%)" if col_i == 0 else "")
    ax_ust.set_ylim(0, max(null_pct) * 1.35 + 5)
    ax_ust.spines[["top", "right"]].set_visible(False)

    # Alt: Çözülme durumu pasta grafiği
    ax_alt = axes[1, col_i]
    coz_n  = (alt["is_resolved"] == "Çözüldü").sum()
    cdeg_n = (alt["is_resolved"] == "Çözülmedi").sum()
    wedges, _, autotexts = ax_alt.pie(
        [coz_n, cdeg_n],
        labels=["Çözüldü", "Çözülmedi"],
        colors=[COZUM_RENK["Çözüldü"], COZUM_RENK["Çözülmedi"]],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
    ax_alt.set_title(f"{BANKA_TR[key]}\nTüm Şikayetler (n={toplam:,})",
                     fontsize=10, fontweight="bold")

fig.text(0.5, -0.02,
         "* Şirket Yanıtı eksik = yanıt yok (veri hatası değil)\n"
         "Eksik Oran Renk Kodu: 🟢 <%5 (İyi)   🟡 %5–20 (Dikkat)   🔴 >%20 (Yüksek)",
         ha="center", fontsize=9, color="gray")

plt.tight_layout()
plt.savefig("results/figures/09_veri_kalite_panel.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 09_veri_kalite_panel.png")

# ===========================================================
# TERMİNAL ÖZET
# ===========================================================
print("\n" + "=" * 65)
print("ÖZET İSTATİSTİKLER")
print("=" * 65)
print(ozet.to_string(index=False))

print("\n" + "=" * 65)
print("ADIM 2 TAMAMLANDI!")
print("=" * 65)
print("""
Üretilen dosyalar:
  data/processed/veri_ham_birlesmis.csv   ← Adım 3 bunu kullanır
  data/processed/ozet_istatistikler.csv
  data/processed/null_raporu.csv
  results/figures/01_sikayet_sayisi.png
  results/figures/02_cozulme_durumu.png
  results/figures/03_satisfaction_dagilimi.png
  results/figures/04_satisfaction_cozum_karsilastirma.png
  results/figures/05_aylik_trend.png
  results/figures/06_sirket_yanit_orani.png
  results/figures/07_goruntulenme_dagilimi.png
  results/figures/08_sikayet_kategorileri.png
  results/figures/09_veri_kalite_panel.png

Sıradaki adım: python adim3_onisleme.py
""")