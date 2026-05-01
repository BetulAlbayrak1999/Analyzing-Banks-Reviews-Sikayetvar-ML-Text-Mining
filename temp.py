import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(BASE_DIR, "data", "raw", "sikayetvar_isbank.csv")

df = pd.read_csv(file_path, encoding="utf-8")
print(df.head())
print(file_path)


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from collections import Counter

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

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
DOSYALAR = {
    "VakifBank":  "data/raw/sikayetvar_vakifbank.csv",
    "IsBank":     "data/raw/sikayetvar_isbank.csv",
    "KuveytTurk": "data/raw/sikayetvar_kuveyt_turk.csv",
}
BANKA_TR = {
    "VakifBank":  "VakıfBank",
    "IsBank":     "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}
RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBank":     "#B71C1C",
    "KuveytTurk": "#1B5E20",
}
BANKA_SIRASI = ["VakifBank", "IsBank", "KuveytTurk"]
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
# FİGÜR
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
EOF
cd /home/claude && python3 fig08.py
Output

✓ 08_sikayet_kategorileri.png kaydedildi

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
