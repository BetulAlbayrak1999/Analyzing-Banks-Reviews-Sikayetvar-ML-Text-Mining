"""
=============================================================
ADIM 3: TURKCE METIN ON ISLEME (NLP Preprocessing)
=============================================================
Bu script:
1. Birlesmis veriyi yukler
2. Turkce metinlere NLP on isleme uygular:
   - Kucuk harfe cevir
   - URL, mention, ozel karakter temizle
   - Noktalama kaldir
   - Sayi kaldir
   - Turkce stop-word cikar
   - Stemming (Snowball - Turkce)
3. Temizlenmis metni kaydeder → data/processed/veri_temiz.csv
4. On isleme oncesi/sonrasi ornekler gosterir
5. Kelime bulutu gorseli uretir

CALISTIRMA: python adim3_onisleme.py

GEREKLI KUTUPHANE:
  pip install snowballstemmer nltk

=============================================================
ADIM 3 (DUZELTILMIS): TURKCE METIN ON ISLEME
=============================================================
DUZELTME:
  Onceki versiyonda [a-z] regex satiri Turkce harfleri
  (o, u, g, s vb.) de siliyordu → metinler bos kaliyordu.
  Bu versiyon yalnizca noktalama ve ozel sembolleri kaldirir,
  Turkce alfabesine dokunmaz.
=============================================================
"""

import pandas as pd
import numpy as np
import re, os, warnings
warnings.filterwarnings("ignore")

import nltk
nltk.download("punkt",    quiet=True)
nltk.download("punkt_tab",quiet=True)
nltk.download("stopwords",quiet=True)
from nltk.corpus import stopwords

from snowballstemmer import TurkishStemmer

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["font.family"]      = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

try:
    from wordcloud import WordCloud
    WORDCLOUD_VAR = True
except ImportError:
    WORDCLOUD_VAR = False
    print("UYARI: pip install wordcloud")

os.makedirs("data/processed",  exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# ----------------------------------------------------------
# 1. STOP WORDS
# ----------------------------------------------------------
NLTK_TR_STOP = set(stopwords.words("turkish"))

OZEL_STOP = {
    # Platform
    "devamini", "gor", "sikayetvar",
    # Banka isimleri (konu modellemede gurultu olusturmamak icin)
    "vakifbank", "isbank", "isbankasi", "kuveytturk", "kuveyt", "turk",
    "banka", "bankasi", "bankamiz", "bankamizia",
    # Sik gecen ama anlamsiz
    "musteri", "musterimiz", "sayin", "bilgi", "bilgilendirme",
    "iletisim", "merkezi", "lutfen", "tesekkur", "ederiz",
    "merhaba", "ilgili", "konu", "talebiniz", "basvurunuz",
    # Zaman (dusuk analitik deger)
    "gun", "bugun", "dun", "hafta", "ay", "yil", "saat", "dakika",
    # Web
    "tr", "com", "http", "https", "www",
}

TURKCE_STOP = NLTK_TR_STOP | OZEL_STOP
print(f"Toplam stop word: {len(TURKCE_STOP)}")

# ----------------------------------------------------------
# 2. STEMMER
# ----------------------------------------------------------
stemmer = TurkishStemmer()

# ----------------------------------------------------------
# 3. TEMIZLEME FONKSIYONU (DUZELTILMIS)
# ----------------------------------------------------------
def turkce_temizle(metin: str, stem: bool = True) -> str:
    """
    Turkce sikayet metnini NLP on islemeden gecirir.

    Adimlar:
      1. None/NaN kontrolu
      2. Kucuk harfe cevir
      3. URL temizle
      4. @mention temizle
      5. Rakam temizle
      6. Sadece noktalama ve ozel semboller kaldir
         (Turkce harflere DOKUNULMAZ - onceki [a-z] hatasi duzeltildi)
      7. Fazla bosluklari temizle
      8. Tokenize (bosluga gore)
      9. Stop-word cikar
      10. Uzunluk filtresi (> 2 karakter)
      11. Stemming (TurkishStemmer) - istege bagli
    """
    if pd.isna(metin) or not isinstance(metin, str):
        return ""

    # Adim 2: Kucuk harf
    metin = metin.lower()

    # Adim 3: URL
    metin = re.sub(r"http\S+|www\.\S+", " ", metin)

    # Adim 4: @mention
    metin = re.sub(r"@\w+", " ", metin)

    # Adim 5: Rakam
    metin = re.sub(r"\d+", " ", metin)

    # Adim 6: Noktalama ve ozel semboller
    # DUZELTME: [a-z] satiri kaldirildi.
    # \w  → harf, rakam, alt cizgi
    # \s  → bosluk
    # Turkce ozel karakterler \w kapsamindadir (unicode modu ile)
    metin = re.sub(r"[^\w\s]", " ", metin, flags=re.UNICODE)

    # Alt cizgi kaldir (\_)
    metin = metin.replace("_", " ")

    # Adim 7: Fazla bosluk
    metin = re.sub(r"\s+", " ", metin).strip()

    # Adim 8: Tokenize
    tokenlar = metin.split()

    # Adim 9: Stop-word cikar + Adim 10: Uzunluk filtresi
    tokenlar = [
        t for t in tokenlar
        if t not in TURKCE_STOP and len(t) > 2
    ]

    # Adim 11: Stemming
    if stem and tokenlar:
        tokenlar = [stemmer.stemWord(t) for t in tokenlar]
        # Stemming sonrasi tekrar kisa token temizle
        tokenlar = [t for t in tokenlar if len(t) > 2]

    return " ".join(tokenlar)


# ----------------------------------------------------------
# 4. HIZLI TEST (calistirmadan once dogrula)
# ----------------------------------------------------------
test_metin = "VakıfBank hesabımdan sigorta primlerim için tanımlı otomatik ödeme talimatı nedeniyle ödeyemiyorum!"
test_sonuc = turkce_temizle(test_metin, stem=True)
print(f"\nHIZLI TEST:")
print(f"  Ham   : {test_metin}")
print(f"  Temiz : {test_sonuc}")
assert len(test_sonuc) > 5, "HATA: Temizleme sonrasi metin bos! Fonksiyonu kontrol et."
print("  Test gecti!\n")

# ----------------------------------------------------------
# 5. VERI YUKLE
# ----------------------------------------------------------
print("Veri yukleniyor...")
veri = pd.read_csv("data/processed/veri_ham_birlesmis.csv", encoding="utf-8-sig")
print(f"  {len(veri)} satir yuklendi")

# ----------------------------------------------------------
# 6. ON ISLEME UYGULA
# ----------------------------------------------------------
print("\nOn isleme uygulanıyor... (1-3 dakika surebilir)")

# Stem UYGULANMIS → ML modelleri / TF-IDF icin
veri["temiz_metin"] = veri["full_text"].apply(
    lambda x: turkce_temizle(x, stem=True)
)

# Stem UYGULANMAMIS → LDA icin (kelime anlami korunur)
veri["token_listesi"] = veri["full_text"].apply(
    lambda x: turkce_temizle(x, stem=False)
)

veri["temiz_kelime_sayisi"] = veri["temiz_metin"].apply(lambda x: len(x.split()))

print("  Tamamlandi!")

# ----------------------------------------------------------
# 7. ONCESI / SONRASI KARSILASTIRMA
# ----------------------------------------------------------
# Filtre: temiz_metin uzunlugu > 20 karakter olan satirlar
uygun = veri[veri["temiz_metin"].str.len() > 20]
print(f"\n  Uzunluk>20 filtresi sonrasi ornek havuzu: {len(uygun)} satir")

if len(uygun) >= 3:
    ornekler = uygun.sample(3, random_state=42)
elif len(uygun) > 0:
    ornekler = uygun.sample(len(uygun), random_state=42)
else:
    ornekler = veri.sample(min(3, len(veri)), random_state=42)

print("\n" + "="*60)
print("ON ISLEME ONCESI / SONRASI ORNEKLER")
print("="*60)
for i, (_, row) in enumerate(ornekler.iterrows(), 1):
    ham   = str(row.get("full_text", ""))[:250]
    temiz = str(row.get("temiz_metin", ""))[:250]
    ham_k = row.get("kelime_sayisi", "?")
    temiz_k = row.get("temiz_kelime_sayisi", "?")
    print(f"\n--- Ornek {i} ({row['banka_label']}) ---")
    print(f"HAM    : {ham}")
    print(f"TEMIZ  : {temiz}")
    print(f"Kelime : {ham_k} → {temiz_k}")

# ----------------------------------------------------------
# 8. ISTATISTIK
# ----------------------------------------------------------
print("\n--- BANKA BAZLI TEMIZLEME ISTATISTIGI ---")
istat = veri.groupby("banka_label").agg(
    ham_ort_kelime   = ("kelime_sayisi",       "mean"),
    temiz_ort_kelime = ("temiz_kelime_sayisi",  "mean"),
    bos_metin_sayisi = ("temiz_metin", lambda x: (x.str.strip() == "").sum()),
).round(1)
print(istat.to_string())

# ----------------------------------------------------------
# 9. BOŞ METİN FİLTRELE
# ----------------------------------------------------------
onceki = len(veri)
veri = veri[veri["temiz_metin"].str.strip().str.len() > 3].copy()
print(f"\nBos metin filtresi: {onceki} → {len(veri)} satir kaldi")

# ----------------------------------------------------------
# 10. KAYDET
# ----------------------------------------------------------
veri.to_csv("data/processed/veri_temiz.csv", index=False, encoding="utf-8-sig")
print(f"  -> veri_temiz.csv kaydedildi ({len(veri)} satir)")

# ----------------------------------------------------------
# 11. KELIME UZUNLUGU KARSILASTIRMA GRAFİGİ
# ----------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
banka_sirasi = ["VakifBank", "IsBankasi", "KuveytTurk"]
RENKLER = {"VakifBank":"#1565C0", "IsBankasi":"#B71C1C", "KuveytTurk":"#1B5E20"}

for ax, (sutun, baslik) in zip(axes, [
    ("kelime_sayisi",       "Ham Kelime Sayısı (Ön İsleme Öncesi)"),
    ("temiz_kelime_sayisi", "Temiz Kelime Sayısı (Ön İsleme Sonrası)"),
]):
    for banka in banka_sirasi:
        alt   = veri[veri["banka_label"] == banka][sutun]
        sinir = int(alt.quantile(0.95))
        ax.hist(alt.clip(upper=sinir), bins=40, alpha=0.6,
                color=RENKLER[banka], label=banka, edgecolor="none")
    ax.set_title(baslik, fontsize=12, fontweight="bold")
    ax.set_xlabel("Kelime Sayısı")
    ax.set_ylabel("Şikayet Adedi")
    ax.legend()

plt.suptitle("Ön İsleme Öncesi vs Sonrası Kelime Sayısı Dağılımı",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/figures/08_onisleme_karsilastirma.png", dpi=150)
plt.close()
print("  -> 08_onisleme_karsilastirma.png kaydedildi")

# ----------------------------------------------------------
# 12. KELIME BULUTU
# ----------------------------------------------------------
if WORDCLOUD_VAR:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    RENK_MAP = {"VakifBank":"Blues", "IsBankasi":"Reds", "KuveytTurk":"Greens"}

    for ax, banka in zip(axes, banka_sirasi):
        alt      = veri[veri["banka_label"] == banka]["temiz_metin"]
        birlesik = " ".join(alt.dropna().tolist())
        if len(birlesik) < 20:
            ax.set_title(f"{banka}\n(Veri yok)"); continue
        wc = WordCloud(
            width=600, height=400,
            background_color="white",
            colormap=RENK_MAP[banka],
            max_words=100,
            collocations=False,
        ).generate(birlesik)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(banka, fontsize=13, fontweight="bold")

    fig.suptitle("En Sık Geçen Kelimeler - Temizlenmiş Metin (Stem Uygulanmış)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/figures/09_kelime_bulutu.png", dpi=150)
    plt.close()
    print("  -> 09_kelime_bulutu.png kaydedildi")

# ----------------------------------------------------------
# OZET
# ----------------------------------------------------------
print("\n" + "="*60)
print("ADIM 3 TAMAMLANDI!")
print()
print("Uretilen dosyalar:")
print("  data/processed/veri_temiz.csv               <- Adim 4 ve 5 bunu kullanir")
print("  results/figures/08_onisleme_karsilastirma.png")
print("  results/figures/09_kelime_bulutu.png")
print()
print("veri_temiz.csv yeni sutunlar:")
print("  temiz_metin        : stem uygulanmis  → ML/TF-IDF icin")
print("  token_listesi      : stem uygulanmamis → LDA icin")
print("  temiz_kelime_sayisi: on isleme sonrasi kelime sayisi")
print()
print("Siradaki adim: python adim4_lda.py")
print("="*60)