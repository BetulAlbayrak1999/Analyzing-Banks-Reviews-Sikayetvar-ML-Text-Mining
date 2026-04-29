"""
=============================================================
ADIM 4: LDA KONU MODELLEME (Topic Modeling)
=============================================================
Bu script ne yapıyor?
Coherence Score ile optimal k bulma:
Script her banka için k=3'ten k=10'a kadar LDA modeli 
eğitip her birinin Coherence Score'unu ölçüyor. 
En yüksek skoru veren k değeri otomatik seçiliyor. 
Grafik terminale şöyle yazıyor:
k=3(0.412) k=4(0.445) k=5(0.461) k=6(0.438) ...
→ Optimal konu sayisi: 5
Her bankaya ayrı model: 3 banka için 3 ayrı LDA modeli
eğitilir çünkü şikayet konuları banka tipine göre 
farklılaşır (katılım bankası vs kamu bankası).

Bu script:
1. Temizlenmis veriyi yukler (veri_temiz.csv)
2. Her banka icin ayri LDA modeli egitir
3. Optimal konu sayisini Coherence Score ile belirler
4. Konu kelimelerini ve dagilimini raporlar
5. pyLDAvis ile interaktif HTML gorseli uretir
6. Her sikayete en yuksek olasilikli konuyu atar

CALISTIRMA: python adim4_lda.py

GEREKLI:
  pip install gensim pyldavis
=============================================================

=============================================================
DUZELTME:
  Windows'ta CoherenceModel multiprocessing hatasini onlemek icin:
  - Tum kod if __name__ == '__main__': blogu altina alindi
  - CoherenceModel processes=1 ile calistirildi (single-thread)
  - Bu degisiklik sonucu etkilemez, sadece Windows uyumlulugu saglar

=============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, warnings, json
warnings.filterwarnings("ignore")
matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["axes.unicode_minus"] = False

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel

try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    PYLDAVIS_VAR = True
except ImportError:
    PYLDAVIS_VAR = False
    print("UYARI: pip install pyldavis  (olmadan da calisir)")

# ----------------------------------------------------------
# PARAMETRELER
# ----------------------------------------------------------
KONU_ARALIK     = range(3, 11)
LDA_PASSES      = 20
LDA_ITERATIONS  = 100
RANDOM_STATE    = 42
MIN_KELIME_FREQ = 3
MAX_KELIME_ORT  = 0.85

BANKA_SIRASI = ["VakifBank", "IsBankasi", "KuveytTurk"]
RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBankasi":  "#B71C1C",
    "KuveytTurk": "#1B5E20",
}

# ----------------------------------------------------------
# YARDIMCI FONKSIYONLAR
# (Fonksiyonlar if __name__ blogu disinda olmali)
# ----------------------------------------------------------

def metin_tokenize(seri):
    return [str(m).split() for m in seri]


def sozluk_ve_korpus_olustur(token_listesi):
    sozluk = corpora.Dictionary(token_listesi)
    sozluk.filter_extremes(no_below=MIN_KELIME_FREQ, no_above=MAX_KELIME_ORT)
    korpus = [sozluk.doc2bow(doc) for doc in token_listesi]
    return sozluk, korpus


def coherence_hesapla(token_listesi, sozluk, korpus, konu_aralik):
    """
    Windows multiprocessing hatasini onlemek icin:
    processes=1  →  tek thread, guvenli
    """
    skorlar = []
    print("    Coherence hesaplaniyor: ", end="", flush=True)
    for k in konu_aralik:
        model = LdaModel(
            corpus=korpus,
            id2word=sozluk,
            num_topics=k,
            passes=10,
            iterations=50,
            random_state=RANDOM_STATE,
            alpha="auto",
            eta="auto",
        )
        cm = CoherenceModel(
            model=model,
            texts=token_listesi,
            dictionary=sozluk,
            coherence="c_v",
            processes=1,        # ← WINDOWS DUZELTMESI: multiprocessing kapali
        )
        skor = cm.get_coherence()
        skorlar.append(skor)
        print(f"k={k}({skor:.3f}) ", end="", flush=True)
    print()
    return skorlar


def lda_egit(korpus, sozluk, num_topics):
    model = LdaModel(
        corpus=korpus,
        id2word=sozluk,
        num_topics=num_topics,
        passes=LDA_PASSES,
        iterations=LDA_ITERATIONS,
        random_state=RANDOM_STATE,
        alpha="auto",
        eta="auto",
        minimum_probability=0.01,
    )
    return model


def konu_etiketle(model, num_topics, topn=10):
    konular = {}
    for k in range(num_topics):
        kelimeler = [w for w, _ in model.show_topic(k, topn=topn)]
        konular[k] = kelimeler
    return konular


def sikayet_konu_ata(model, korpus):
    atamalar = []
    for doc_bow in korpus:
        dagilim = model.get_document_topics(doc_bow, minimum_probability=0)
        en_yuksek = max(dagilim, key=lambda x: x[1])
        atamalar.append(en_yuksek[0])
    return atamalar


# ===========================================================
# ANA PROGRAM — Windows icin zorunlu
# ===========================================================
if __name__ == "__main__":

    os.makedirs("results/lda",     exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("models",          exist_ok=True)
    os.makedirs("data/processed",  exist_ok=True)

    # --------------------------------------------------------
    # 1. VERI YUKLE
    # --------------------------------------------------------
    print("Veri yukleniyor...")
    veri = pd.read_csv("data/processed/veri_temiz.csv", encoding="utf-8-sig")
    print(f"  {len(veri)} satir yuklendi")

    # token_listesi kontrolu
    if "token_listesi" not in veri.columns:
        print("  UYARI: token_listesi yok, temiz_metin kullaniliyor")
        veri["token_listesi"] = veri["temiz_metin"]

    veri = veri[
        veri["token_listesi"].notna() &
        (veri["token_listesi"].str.strip() != "")
    ].copy()
    print(f"  Gecerli satir: {len(veri)}")

    tum_sonuclar = {}

    # --------------------------------------------------------
    # 2. HER BANKA ICIN LDA
    # --------------------------------------------------------
    for banka in BANKA_SIRASI:
        print(f"\n{'='*55}")
        print(f"  BANKA: {banka}")
        print(f"{'='*55}")

        alt = veri[veri["banka_label"] == banka].copy()
        print(f"  Sikayet sayisi: {len(alt)}")

        if len(alt) < 20:
            print(f"  UYARI: Yeterli veri yok, {banka} atlaniyor!")
            continue

        # Tokenize
        token_listesi = metin_tokenize(alt["token_listesi"])
        token_listesi = [t for t in token_listesi if len(t) >= 3]

        # Sozluk + korpus
        sozluk, korpus = sozluk_ve_korpus_olustur(token_listesi)
        print(f"  Sozluk: {len(sozluk)} kelime | Korpus: {len(korpus)} belge")

        if len(sozluk) < 10:
            print(f"  UYARI: Sozluk cok kucuk, atlaniyor!")
            continue

        # Coherence ile optimal k
        print(f"  Optimal konu sayisi aranıyor...")
        coherence_skorlar = coherence_hesapla(
            token_listesi, sozluk, korpus, KONU_ARALIK
        )

        optimal_k = list(KONU_ARALIK)[np.argmax(coherence_skorlar)]
        en_yuksek_skor = max(coherence_skorlar)
        print(f"  -> Optimal k={optimal_k}  Coherence={en_yuksek_skor:.4f}")

        # Coherence grafigi
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(list(KONU_ARALIK), coherence_skorlar,
                marker="o", linewidth=2, color=RENKLER[banka])
        ax.axvline(optimal_k, color="red", linestyle="--",
                   linewidth=1.5, label=f"Optimal: k={optimal_k}")
        ax.set_title(f"{banka} - Coherence Score vs Konu Sayisi",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Konu Sayisi (k)")
        ax.set_ylabel("Coherence Score (c_v)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(f"results/figures/lda_coherence_{banka.lower()}.png", dpi=150)
        plt.close()
        print(f"  -> Coherence grafigi kaydedildi")

        # Final LDA
        print(f"  Final LDA egitiliyor (k={optimal_k}, passes={LDA_PASSES})...")
        model = lda_egit(korpus, sozluk, optimal_k)

        # Konulari yazdir
        konular = konu_etiketle(model, optimal_k, topn=10)
        print(f"\n  --- KONULAR ---")
        for k, kelimeler in konular.items():
            print(f"  Konu {k+1:2d}: {', '.join(kelimeler)}")

        # Konu atama
        alt_indeks = alt.index.tolist()[:len(korpus)]
        atamalar   = sikayet_konu_ata(model, korpus)
        veri.loc[alt_indeks, f"lda_konu_{banka}"] = atamalar

        # Konu dagilim grafigi
        konu_sayilari = pd.Series(atamalar).value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(9, 4))
        etiketler = [f"Konu {i+1}" for i in konu_sayilari.index]
        bars = ax.bar(etiketler, konu_sayilari.values,
                      color=RENKLER[banka], edgecolor="white")
        for bar, val in zip(bars, konu_sayilari.values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 3, str(val),
                    ha="center", fontsize=9)
        ax.set_title(f"{banka} - Konulara Gore Sikayet Dagilimi",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Konu")
        ax.set_ylabel("Sikayet Sayisi")
        plt.tight_layout()
        plt.savefig(f"results/figures/lda_dagilim_{banka.lower()}.png", dpi=150)
        plt.close()
        print(f"  -> Dagilim grafigi kaydedildi")

        # pyLDAvis HTML
        if PYLDAVIS_VAR:
            try:
                vis = gensimvis.prepare(model, korpus, sozluk, sort_topics=False)
                html_yol = f"results/lda/ldavis_{banka.lower()}.html"
                pyLDAvis.save_html(vis, html_yol)
                print(f"  -> pyLDAvis HTML: {html_yol}  (tarayicida ac)")
            except Exception as e:
                print(f"  UYARI: pyLDAvis hatasi: {e}")

        # Model kaydet
        model.save(f"models/lda_{banka.lower()}.model")
        print(f"  -> Model kaydedildi: models/lda_{banka.lower()}.model")

        # Sonuclari sakla
        tum_sonuclar[banka] = {
            "optimal_k":         optimal_k,
            "coherence_max":     round(en_yuksek_skor, 4),
            "coherence_skorlar": {
                str(k): round(s, 4)
                for k, s in zip(KONU_ARALIK, coherence_skorlar)
            },
            "konular": {
                f"Konu_{k+1}": v for k, v in konular.items()
            },
            "sikayet_sayisi": len(alt),
            "sozluk_boyutu":  len(sozluk),
        }

    # --------------------------------------------------------
    # 3. KARSILASTIRMALI COHERENCE GRAFİGİ
    # --------------------------------------------------------
    if tum_sonuclar:
        fig, ax = plt.subplots(figsize=(10, 5))
        for banka in BANKA_SIRASI:
            if banka not in tum_sonuclar:
                continue
            skorlar = list(tum_sonuclar[banka]["coherence_skorlar"].values())
            ax.plot(list(KONU_ARALIK), skorlar,
                    marker="o", linewidth=2,
                    color=RENKLER[banka], label=banka)
            opt     = tum_sonuclar[banka]["optimal_k"]
            opt_idx = list(KONU_ARALIK).index(opt)
            ax.annotate(
                f"k={opt}",
                xy=(opt, skorlar[opt_idx]),
                xytext=(opt + 0.15, skorlar[opt_idx] + 0.003),
                fontsize=9, color=RENKLER[banka],
            )
        ax.set_title("Coherence Score Karsilastirmasi (3 Banka)",
                     fontsize=13, fontweight="bold")
        ax.set_xlabel("Konu Sayisi (k)")
        ax.set_ylabel("Coherence Score (c_v)")
        ax.legend()
        plt.tight_layout()
        plt.savefig("results/figures/lda_coherence_karsilastirma.png", dpi=150)
        plt.close()
        print("\n-> Karsilastirmali coherence grafigi kaydedildi")

    # --------------------------------------------------------
    # 4. SONUCLARI KAYDET
    # --------------------------------------------------------
    # JSON
    with open("results/lda/lda_sonuclar.json", "w", encoding="utf-8") as f:
        json.dump(tum_sonuclar, f, ensure_ascii=False, indent=2)
    print("-> results/lda/lda_sonuclar.json kaydedildi")

    # Konu-Kelime CSV tablosu
    satirlar = []
    for banka, sonuc in tum_sonuclar.items():
        for konu_adi, kelimeler in sonuc["konular"].items():
            satirlar.append({
                "Banka":     banka,
                "Konu":      konu_adi,
                "Kelimeler": ", ".join(kelimeler),
                "Optimal_k": sonuc["optimal_k"],
                "Coherence": sonuc["coherence_max"],
            })
    konu_df = pd.DataFrame(satirlar)
    konu_df.to_csv("results/lda/konu_kelime_tablosu.csv",
                   index=False, encoding="utf-8-sig")
    print("-> results/lda/konu_kelime_tablosu.csv kaydedildi")

    # Konu atamali veri seti
    veri.to_csv("data/processed/veri_lda.csv",
                index=False, encoding="utf-8-sig")
    print("-> data/processed/veri_lda.csv kaydedildi")

    # --------------------------------------------------------
    # OZET
    # --------------------------------------------------------
    print("\n" + "="*60)
    print("ADIM 4 TAMAMLANDI!")
    print()
    if tum_sonuclar:
        print("Optimal Konu Sayilari:")
        for banka, sonuc in tum_sonuclar.items():
            print(f"  {banka:12s}: k={sonuc['optimal_k']}  "
                  f"Coherence={sonuc['coherence_max']}")
    print()
    print("Onemli dosyalar:")
    print("  results/lda/ldavis_*.html           <- tarayicida ac")
    print("  results/lda/konu_kelime_tablosu.csv <- tez tablosu")
    print("  data/processed/veri_lda.csv         <- Adim 5 bunu kullanir")
    print()
    print("Siradaki adim: python adim5_tfidf_ml.py")
    print("="*60)