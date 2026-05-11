"""
=============================================================
ADIM 4 (UZMAN): LDA KONU MODELLEME (Topic Modeling)
=============================================================
Bankacılık Sektörü Müşteri Şikayet Analizi — 2025
Kuveyt Türk | VakıfBank | İşBankası

TEMEL TASARIM KARARLARI:
  ✓ LDA input: temiz_metin (stem + post-stem filtreli)
    → token_listesi yerine temiz_metin tercih edildi:
       token_listesi'nde çekimli form gürültüsü var
       (şekilde, tarafıma, üzerinden vb.)
    → temiz_metin: 695 + 69 stop-word = 764 toplam filtre
  ✓ LDA-özel ek filtre (post-stem'den kaçanlar):
    kan, yapamıyor, arkadaş, kapsam, açık, bilgi,
    dönüş, sorun, sıra, numaral, maximum, kazandıra ...
  ✓ Her banka için ayrı coherence taraması (k=3..9)
  ✓ Konu etiketleri: bankacılık domain bilgisiyle manuel
  ✓ Bigram tespiti (Gensim Phrases): "kredi_kartı",
    "mobil_uygulama", "ihtiyaç_kredisi" gibi birleşik terimler
  ✓ Perplexity + Coherence birlikte raporlanır
  ✓ Konu-çözüm ilişkisi: her konu için çözülme oranı
  ✓ 6 figür: coherence, konu dağılımı, konu-çözüm ısı haritası,
    konu zaman trendi, konu kelime bubble chart, pyLDAvis HTML

ÜRETİLEN ÇIKTILAR:
  models/lda_{banka}.model
  models/lda_sozluk_{banka}.gensim
  results/lda/lda_sonuclar.json
  results/lda/konu_kelime_tablosu.csv
  results/lda/konu_cozum_analizi.csv
  data/processed/veri_lda.csv
  results/figures/17_lda_coherence.png
  results/figures/18_lda_konu_dagilimi.png
  results/figures/19_lda_konu_cozum_isi.png
  results/figures/20_lda_konu_trend.png
  results/figures/21_lda_konu_kelime_bubble.png
  results/lda/ldavis_{banka}.html  (tarayıcıda açın)

ÇALIŞTIRMA: python adim4_lda.py
=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as mcolors

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 150

import pandas as pd
import numpy as np
import os, warnings, json
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")

from gensim import corpora
from gensim.models import LdaModel, CoherenceModel, Phrases
from gensim.models.phrases import Phraser

try:
    import pyLDAvis
    import pyLDAvis.gensim_models as gensimvis
    PYLDAVIS_VAR = True
except ImportError:
    PYLDAVIS_VAR = False
    print("UYARI: pip install pyldavis")

os.makedirs("results/lda",     exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("models",          exist_ok=True)
os.makedirs("data/processed",  exist_ok=True)

# ============================================================
# SABİTLER
# ============================================================
BANKA_SIRASI = ["VakifBank", "IsBank", "KuveytTurk"]
BANKA_TR = {
    "VakifBank":  "VakıfBank",
    "IsBank":     "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}
BANKA_LABEL = {
    "VakifBank":  "VakıfBank",
    "IsBank":     "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}
RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBank":     "#C62828",
    "KuveytTurk": "#2E7D32",
}

# LDA parametreleri
KONU_ARALIK    = range(3, 10)   # k=3..9
LDA_PASSES     = 20
LDA_ITERATIONS = 150
RANDOM_STATE   = 42
MIN_FREQ       = 5              # sözlükte min belge frekansı
MAX_ORAN       = 0.80           # max belge oranı

# ============================================================
# LDA-ÖZEL EK FİLTRE
# ============================================================
# temiz_metin'den post-stem filtresini kaçırmış formlar
# + bankacılık LDA'sında analitik değeri olmayan genel kelimeler
LDA_EXTRA_STOP = {
    # Post-stem gürültü
    "kan",          # kanun → kan
    "yapamıyor",    # eylem
    "arkadaş",      # referans programı kaynaklı ama gürültü
    "kap",          # kapsamında → kap
    "açık",         # bağlam bağımsız sıfat
    "bilgi",        # genel terim
    "gerekli",      # gerekli → gerekli (sıfat, analitik değersiz)
    "geri",         # geri → ger (stop'ta) ama ham hali
    "dönüş",        # geri dönüş → platform kalıbı
    "sorun",        # sorun → sor (stop'ta) ama ham hali
    "sıra",         # sırasında → sıra
    "numaral",      # numaralı → numaral
    "maximum",      # marka adı
    "maxim",        # stem bozukluğu
    "kazandıra",    # kazandıran → stem bozukluğu
    "kazandıran",   # VakıfBank tarife ürünü, LDA'da gürültü
    "söyle",        # eylem
    "söyledi",      # eylem
    "iletiş",       # iletişim → stop'ta ama stem bozuk hali
    "giderilme",    # pasif eylem
    "yapılmadı",    # eylem
    "promosyo",     # promosyon → stem bozukluğu
    "tesl",         # teslim → stem bozukluğu
    "ediç",         # stem bozukluğu
    "müşteris",     # müşterisi → stop'ta ama kaçmış
    "dikka",        # dikkat → stem bozukluğu
    "gitme",        # eylem
    "yazıl",        # yazıldı → pasif eylem
    "yapıla",       # yapıldı → pasif eylem
    "başvurus",     # başvurusu → genel terim
    "telefo",       # telefon → stem bozukluğu
    "siste",        # sistem → stem bozukluğu
    "fizik",        # fiziksel → sıfat, bağlam bağımsız
    "transfer",     # genel terim (havale/eft ile aynı anlam)
    "otomatik",     # sıfat
    "süreç",        # genel terim
    "dosya",        # genel terim
    "açıkla",       # eylem
    "sayıl",        # sayılı → bağlam bağımsız
    "haksız",       # sıfat — anlamlı ama gürültü
    "talep",        # platform kalıbı zaten stop'ta
    "düzenli",      # sıfat
    "tarih",        # zaman ifadesi — stop'ta ama geçmiş olabilir
    "tarih",
}

# ============================================================
# KONU ETİKETLERİ
# ============================================================
# Her banka için beklenen konu etiketleri (bankacılık domain bilgisi)
# Coherence sonuçlarına göre güncellenir

KONU_ETIKETI_TAHMINI = {
    "VakifBank": {
        0: "ATM, Alışveriş ve Kampanya",
        1: "Kredi ve Borç Yönetimi",
        2: "Dijital Bankacılık ve Güvenlik",
        3: "Kart Aidat ve Ücret Şikayetleri",
        4: "Sigorta ve İhtiyaç Kredisi",
    },
    "IsBank": {
        0: "Kredi Kartı Limiti ve Kampanya",
        1: "Mobil Uygulama ve İşlem Sorunları",
        2: "Aidat ve Yıllık Ücret İtirazı",
        3: "Maaş Promosyonu ve Kart Teslimatı",
        4: "Borç, Haciz ve Hesap Blokesi",
    },
    "KuveytTurk": {
        0: "Kredi Kartı ve Kampanya Şikayetleri",
        1: "Davet Kampanyası ve ATM/Ödül",
        2: "Para Transferi ve Şube İşlemleri",
        3: "Müşteri Hizmetleri ve Görüşme",
        4: "Mobil Uygulama ve Güvenlik",
        5: "Kart Başvurusu ve Şifre Sorunları",
        6: "Hesap ve Para Yatırma İşlemleri",
    },
}

# ============================================================
# YARDIMCI FONKSİYONLAR
# ============================================================

def lda_tokenize(metin_serisi, banka_key):
    """temiz_metin → LDA için token listesi (extra stop + bigram)"""
    tokenlar = []
    for m in metin_serisi.dropna():
        toks = [
            t for t in str(m).split()
            if t not in LDA_EXTRA_STOP and len(t) >= 3
        ]
        if len(toks) >= 3:
            tokenlar.append(toks)
    return tokenlar


def bigram_ekle(tokenlar, min_count=15, threshold=10):
    """Sık birlikte geçen kelimeleri birleştir: kredi kart → kredi_kart"""
    phrases = Phrases(tokenlar, min_count=min_count, threshold=threshold)
    phraser = Phraser(phrases)
    return [phraser[doc] for doc in tokenlar], phraser


def sozluk_kur(tokenlar):
    sozluk = corpora.Dictionary(tokenlar)
    sozluk.filter_extremes(no_below=MIN_FREQ, no_above=MAX_ORAN)
    sozluk.compactify()
    korpus = [sozluk.doc2bow(doc) for doc in tokenlar]
    return sozluk, korpus


def coherence_tara(tokenlar, sozluk, korpus, aralik):
    """k=3..9 için coherence taraması (tek thread, Windows uyumlu)"""
    skorlar = []
    print("    Tarama: ", end="", flush=True)
    for k in aralik:
        model = LdaModel(
            corpus=korpus, id2word=sozluk,
            num_topics=k, passes=10, iterations=75,
            random_state=RANDOM_STATE, alpha="auto", eta="auto",
        )
        coh = CoherenceModel(
            model=model, texts=tokenlar, dictionary=sozluk,
            coherence="c_v", processes=1,
        ).get_coherence()
        skorlar.append((k, coh))
        print(f"k{k}={coh:.3f} ", end="", flush=True)
    print()
    return skorlar


def lda_egit(korpus, sozluk, num_topics):
    return LdaModel(
        corpus=korpus, id2word=sozluk,
        num_topics=num_topics,
        passes=LDA_PASSES, iterations=LDA_ITERATIONS,
        random_state=RANDOM_STATE,
        alpha="auto", eta="auto",
        minimum_probability=0.01,
    )


def konu_ata(model, korpus):
    """Her belgeye en yüksek olasılıklı konuyu ata"""
    atamalar, dagilimlar = [], []
    for bow in korpus:
        dist = model.get_document_topics(bow, minimum_probability=0)
        dist_sorted = sorted(dist, key=lambda x: x[1], reverse=True)
        atamalar.append(dist_sorted[0][0])
        dagilimlar.append({k: round(v, 4) for k, v in dist_sorted})
    return atamalar, dagilimlar


def konu_etiketle(model, num_topics, topn=10):
    return {k: [w for w, _ in model.show_topic(k, topn=topn)]
            for k in range(num_topics)}


# ============================================================
# ANA PROGRAM
# ============================================================
if __name__ == "__main__":

    print("=" * 65)
    print("ADIM 4 (UZMAN): LDA KONU MODELLEME")
    print("=" * 65)

    # ----------------------------------------------------------
    # 1. VERİ YÜKLE
    # ----------------------------------------------------------
    print("\n[Veri yükleniyor...]")
    if os.path.exists("data/processed/veri_temiz.csv"):
        veri = pd.read_csv("data/processed/veri_temiz.csv",
                           encoding="utf-8-sig")
    else:
        raise FileNotFoundError("data/processed/veri_temiz.csv bulunamadı! "
                                "Önce adim3_onisleme.py çalıştırın.")

    veri["date"] = pd.to_datetime(veri["date"], errors="coerce")
    veri["ay_str"] = veri["date"].dt.strftime("%Y-%m")
    print(f"  {len(veri):,} satır yüklendi")

    tum_sonuclar = {}
    coherence_tablo = []   # karşılaştırmalı grafik için

    # ----------------------------------------------------------
    # 2. HER BANKA İÇİN LDA
    # ----------------------------------------------------------
    for banka in BANKA_SIRASI:
        bl = BANKA_TR[banka]
        print(f"\n{'='*60}")
        print(f"  {bl}")
        print(f"{'='*60}")

        alt = veri[veri["banka_label"] == bl].copy()
        print(f"  Şikayet sayısı: {len(alt):,}")

        if len(alt) < 50:
            print("  ⚠ Yeterli veri yok, atlanıyor!")
            continue

        # Tokenize + Bigram
        print("  Tokenize ve bigram tespiti...")
        tokenlar = lda_tokenize(alt["temiz_metin"], banka)
        tokenlar, phraser = bigram_ekle(tokenlar)

        # Kaç bigram bulundu?
        bigram_sayisi = sum(
            1 for doc in tokenlar for t in doc if "_" in t)
        print(f"  Bigram içeren token: {bigram_sayisi:,}")

        # Sözlük + Korpus
        sozluk, korpus = sozluk_kur(tokenlar)
        print(f"  Sözlük: {len(sozluk):,} terim | "
              f"Korpus: {len(korpus):,} belge")

        if len(sozluk) < 20:
            print("  ⚠ Sözlük çok küçük, atlanıyor!")
            continue

        # Coherence taraması
        print("  Coherence taraması (k=3..9)...")
        coh_skorlar = coherence_tara(tokenlar, sozluk, korpus, KONU_ARALIK)

        # Her banka için coherence kaydet
        for k, c in coh_skorlar:
            coherence_tablo.append({"Banka": banka, "k": k,
                                    "Coherence": c})

        optimal_k   = max(coh_skorlar, key=lambda x: x[1])[0]
        optimal_coh = max(coh_skorlar, key=lambda x: x[1])[1]
        print(f"  ✓ Optimal: k={optimal_k}  "
              f"Coherence={optimal_coh:.4f}")

        # Final model
        print(f"  Final LDA eğitiliyor "
              f"(k={optimal_k}, passes={LDA_PASSES})...")
        model = lda_egit(korpus, sozluk, optimal_k)

        # Perplexity
        perplexity = model.log_perplexity(korpus)
        print(f"  Perplexity (log): {perplexity:.4f}")

        # Konu kelimeleri
        konular = konu_etiketle(model, optimal_k, topn=12)
        print(f"\n  {'─'*50}")
        print("  KONU KELİMELERİ:")
        for k_idx, kelimeler in konular.items():
            tahmin = KONU_ETIKETI_TAHMINI.get(banka, {}).get(
                k_idx, f"Konu {k_idx+1}")
            print(f"  [{k_idx+1:2d}] {tahmin:35s} → "
                  f"{', '.join(kelimeler[:8])}")

        # Konu atama
        atamalar, dagilimlar = konu_ata(model, korpus)

        # veri'ye konu sütunu ekle
        gec_idx = alt.index[:len(korpus)]
        veri.loc[gec_idx, f"lda_konu_{banka}"] = atamalar

        # Konu dağılımı
        konu_dagilim = Counter(atamalar)
        print(f"\n  Konu Dağılımı:")
        for k_idx in range(optimal_k):
            n = konu_dagilim.get(k_idx, 0)
            pct = n / len(atamalar) * 100
            tahmin = KONU_ETIKETI_TAHMINI.get(banka, {}).get(
                k_idx, f"Konu {k_idx+1}")
            print(f"    Konu {k_idx+1:2d} ({tahmin[:30]:30s}): "
                  f"{n:4d} şikayet (%{pct:.1f})")

        # Konu × Çözüm oranı
        alt_kopya = alt.iloc[:len(korpus)].copy()
        alt_kopya["lda_konu"] = atamalar
        konu_cozum = alt_kopya.groupby("lda_konu").agg(
            Sikayet_Sayisi=("is_resolved", "count"),
            Cozulme_Orani=("is_resolved",
                           lambda x: (x == "Çözüldü").mean() * 100),
            Satisfaction_Med=("satisfaction", "median"),
        ).round(2)
        print(f"\n  Konu × Çözüm Analizi:")
        print(konu_cozum.to_string())

        # Kaydet
        model.save(f"models/lda_{banka.lower()}.model")
        sozluk.save(f"models/lda_sozluk_{banka.lower()}.gensim")

        # pyLDAvis
        if PYLDAVIS_VAR:
            try:
                vis = gensimvis.prepare(
                    model, korpus, sozluk, sort_topics=False)
                html_path = f"results/lda/ldavis_{banka.lower()}.html"
                pyLDAvis.save_html(vis, html_path)
                print(f"\n  ✓ pyLDAvis → {html_path}")
            except Exception as e:
                print(f"  ⚠ pyLDAvis hatası: {e}")

        tum_sonuclar[banka] = {
            "banka_tr":         bl,
            "optimal_k":        optimal_k,
            "coherence_max":    round(optimal_coh, 4),
            "perplexity":       round(perplexity, 4),
            "coherence_tum":    {k: round(c, 4) for k, c in coh_skorlar},
            "konular":          {
                f"Konu_{i+1}": {
                    "kelimeler": v,
                    "etiket": KONU_ETIKETI_TAHMINI.get(
                        banka, {}).get(i, f"Konu {i+1}"),
                }
                for i, v in konular.items()
            },
            "konu_dagilim":     {k: v for k, v in konu_dagilim.items()},
            "konu_cozum":       konu_cozum.to_dict(),
            "sikayet_sayisi":   len(alt),
            "sozluk_boyutu":    len(sozluk),
            "bigram_token":     bigram_sayisi,
        }

    # ----------------------------------------------------------
    # 3. KAYDEDILEN VERİ SETİ
    # ----------------------------------------------------------
    veri.to_csv("data/processed/veri_lda.csv",
                index=False, encoding="utf-8-sig")
    print(f"\n✓ data/processed/veri_lda.csv kaydedildi")

    # JSON
    with open("results/lda/lda_sonuclar.json", "w",
              encoding="utf-8") as f:
        json.dump(tum_sonuclar, f, ensure_ascii=False, indent=2)

    # Konu-Kelime CSV
    satirlar = []
    for banka, sonuc in tum_sonuclar.items():
        for konu_adi, konu_bilgi in sonuc["konular"].items():
            satirlar.append({
                "Banka":       BANKA_TR[banka],
                "Konu":        konu_adi,
                "Etiket":      konu_bilgi["etiket"],
                "Kelimeler":   ", ".join(konu_bilgi["kelimeler"]),
                "Optimal_k":   sonuc["optimal_k"],
                "Coherence":   sonuc["coherence_max"],
                "Perplexity":  sonuc["perplexity"],
            })
    pd.DataFrame(satirlar).to_csv(
        "results/lda/konu_kelime_tablosu.csv",
        index=False, encoding="utf-8-sig")

    # ============================================================
    # 4. FİGÜRLER
    # ============================================================
    print("\n[Figürler oluşturuluyor...]")

    coh_df = pd.DataFrame(coherence_tablo)

    # ─── FİGÜR 17: Coherence Karşılaştırma ───────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "LDA Konu Sayısı Optimizasyonu: Coherence Skoru\n"
        "Her banka için k=3..9 arasında en yüksek coherence değeri optimal konu sayısıdır",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, banka in zip(axes, BANKA_SIRASI):
        bl   = BANKA_TR[banka]
        renk = RENKLER[banka]
        sub  = coh_df[coh_df["Banka"] == banka].sort_values("k")
        if sub.empty:
            continue

        ax.set_facecolor("white")
        ax.plot(sub["k"], sub["Coherence"],
                marker="o", lw=2.5, ms=7, color=renk)
        ax.fill_between(sub["k"], sub["Coherence"],
                        alpha=0.10, color=renk)

        opt_row = sub.loc[sub["Coherence"].idxmax()]
        opt_k   = int(opt_row["k"])
        opt_c   = opt_row["Coherence"]

        ax.axvline(opt_k, color=renk, ls="--", lw=1.8, alpha=0.7)
        ax.scatter([opt_k], [opt_c], s=130, color=renk,
                   zorder=5, edgecolors="white", lw=2)
        ax.annotate(
            f" Optimal\n k={opt_k}\n C={opt_c:.3f}",
            xy=(opt_k, opt_c),
            xytext=(opt_k + 0.3, opt_c - 0.015),
            fontsize=9, color=renk, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=renk, lw=1.2),
        )

        ax.set_title(bl, fontsize=12, fontweight="bold", color=renk)
        ax.set_xlabel("Konu Sayısı (k)", fontsize=10)
        ax.set_ylabel("Coherence Skoru (c_v)" if ax == axes[0] else "",
                       fontsize=10)
        ax.set_xticks(list(KONU_ARALIK))
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, ls="--")
        ax.set_ylim(sub["Coherence"].min() - 0.03,
                    sub["Coherence"].max() + 0.05)

        perp = tum_sonuclar.get(banka, {}).get("perplexity", "—")
        ax.text(0.03, 0.04,
                f"Perplexity (log): {perp}",
                transform=ax.transAxes, fontsize=8.5,
                color="#555", style="italic",
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#F5F5F5", alpha=0.9))

    plt.tight_layout()
    plt.savefig("results/figures/17_lda_coherence.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 17_lda_coherence.png")

    # ─── FİGÜR 18: Konu Dağılımı ─────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "LDA Konu Dağılımı: Her Bankada Şikayetlerin Konulara Göre Dağılımı\n"
        "Etiketler: Konu numarası ve anahtar kelimeler  |  Yüzde: toplam içindeki pay",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, banka in zip(axes, BANKA_SIRASI):
        if banka not in tum_sonuclar:
            continue
        bl   = BANKA_TR[banka]
        renk = RENKLER[banka]
        sonuc = tum_sonuclar[banka]

        konu_n     = sonuc["optimal_k"]
        dagilim    = sonuc["konu_dagilim"]
        konular_d  = sonuc["konular"]

        siralama = sorted(dagilim.items(), key=lambda x: x[1],
                          reverse=True)
        k_idx_list = [s[0] for s in siralama]
        sayilar    = [s[1] for s in siralama]
        toplam     = sum(sayilar)

        # Etiket: ilk 3 kelime
        etiketler = []
        for k_idx in k_idx_list:
            konu_adi  = f"Konu_{k_idx+1}"
            bilgi     = konular_d.get(konu_adi, {})
            kelimeler = bilgi.get("kelimeler", [])[:3]
            etiket    = bilgi.get("etiket", f"Konu {k_idx+1}")
            kisa      = "\n".join([f"K{k_idx+1}: {etiket[:22]}",
                                   f"({', '.join(kelimeler)})"])
            etiketler.append(kisa)

        # Renk gradyanı
        base_rgb = mcolors.hex2color(renk)
        renk_listesi = [
            tuple(min(1, c + (1-c)*i/max(konu_n-1,1)*0.6)
                  for c in base_rgb)
            for i in range(len(sayilar))
        ]

        ax.set_facecolor("white")
        bars = ax.barh(range(len(siralama)), sayilar,
                       color=renk_listesi, edgecolor="white",
                       height=0.70)

        for i, (bar, val) in enumerate(zip(bars, sayilar)):
            pct = val / toplam * 100
            ax.text(val + toplam * 0.01, i,
                    f"{val:,}  (%{pct:.1f})",
                    va="center", fontsize=9, fontweight="bold")

        ax.set_yticks(range(len(siralama)))
        ax.set_yticklabels(etiketler, fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(f"{bl}  (k={konu_n}  "
                     f"C={sonuc['coherence_max']:.3f})",
                     fontsize=11, fontweight="bold", color=renk)
        ax.set_xlabel("Şikayet Sayısı", fontsize=10)
        ax.set_xlim(0, max(sayilar) * 1.30)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

    plt.tight_layout()
    plt.savefig("results/figures/18_lda_konu_dagilimi.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 18_lda_konu_dagilimi.png")

    # ─── FİGÜR 19: Konu × Çözüm Isı Haritası ────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Konu × Çözüm Ilişkisi Isı Haritası\n"
        "Her konu için çözülme oranı (%): Koyu kırmızı = düşük çözüm, Koyu yeşil = yüksek çözüm",
        fontsize=13, fontweight="bold", y=1.01
    )

    for ax, banka in zip(axes, BANKA_SIRASI):
        if banka not in tum_sonuclar:
            continue
        bl    = BANKA_TR[banka]
        renk  = RENKLER[banka]
        sonuc = tum_sonuclar[banka]

        # Gerçek veriden konu × çözüm hesapla
        konu_sut = f"lda_konu_{banka}"
        if konu_sut not in veri.columns:
            ax.text(0.5, 0.5, "Konu ataması yok",
                    ha="center", transform=ax.transAxes)
            continue

        alt = veri[veri["banka_label"] == bl].dropna(
            subset=[konu_sut])
        alt["konu_int"] = alt[konu_sut].astype(int)

        konu_n = sonuc["optimal_k"]
        konular_d = sonuc["konular"]

        # Her konu için metrikler
        metrik_veri = []
        for k_idx in range(konu_n):
            k_alt = alt[alt["konu_int"] == k_idx]
            n_toplam = len(k_alt)
            if n_toplam == 0:
                continue
            coz_oran = (k_alt["is_resolved"] == "Çözüldü").mean() * 100
            sat_med  = k_alt["satisfaction"].median()
            konu_adi = f"Konu_{k_idx+1}"
            etiket   = konular_d.get(konu_adi, {}).get(
                "etiket", f"Konu {k_idx+1}")
            metrik_veri.append({
                "konu_idx":   k_idx,
                "etiket":     etiket[:28],
                "sikayet_n":  n_toplam,
                "coz_oran":   coz_oran,
                "sat_med":    sat_med,
            })

        if not metrik_veri:
            continue

        df_m = pd.DataFrame(metrik_veri).sort_values(
            "coz_oran", ascending=True)

        # Isı haritası renk skalası: kırmızı→yeşil
        cmap = plt.cm.RdYlGn
        norm = mcolors.Normalize(vmin=20, vmax=80)

        ax.set_facecolor("white")
        for i, row in enumerate(df_m.itertuples()):
            renk_bar = cmap(norm(row.coz_oran))
            ax.barh(i, row.coz_oran, color=renk_bar,
                    edgecolor="white", height=0.68)
            ax.text(row.coz_oran + 1, i,
                    f"%{row.coz_oran:.1f}  (n={row.sikayet_n:,})",
                    va="center", fontsize=9, fontweight="bold")

            # Memnuniyet bilgisi (sağ taraf)
            sat_txt = (f"★{row.sat_med:.0f}"
                       if not np.isnan(row.sat_med) else "—")
            ax.text(-1, i, sat_txt,
                    va="center", ha="right", fontsize=9,
                    color="#555")

        ax.set_yticks(range(len(df_m)))
        ax.set_yticklabels(
            [f"K{r.konu_idx+1}: {r.etiket}" for r in df_m.itertuples()],
            fontsize=8.5)
        ax.invert_yaxis()
        ax.set_title(f"{bl}\nKonu Bazlı Çözüm Oranı",
                     fontsize=11, fontweight="bold", color=renk)
        ax.set_xlabel("Çözülme Oranı (%)", fontsize=10)
        ax.set_xlim(-5, 105)
        ax.axvline(50, color="#AAA", ls="--", lw=1.2, alpha=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

        # Renk çubuğu
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, shrink=0.5, pad=0.02)
        cbar.set_label("Çözüm %", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

        # ★ notu
        ax.text(0.01, -0.09, "★: Medyan Memnuniyet Skoru (1-5)",
                transform=ax.transAxes, fontsize=7.5, color="#777",
                style="italic")

    plt.tight_layout()
    plt.savefig("results/figures/19_lda_konu_cozum_isi.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 19_lda_konu_cozum_isi.png")

    # ─── FİGÜR 20: Konu Zaman Trendi ─────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Şikayet Konularının Aylık Zaman Trendi\n"
        "Her konu rengi bir şikayet kategorisini temsil eder",
        fontsize=13, fontweight="bold", y=1.01
    )

    AY_TR = {
        "2025-01": "Oca", "2025-02": "Şub", "2025-03": "Mar",
        "2025-04": "Nis", "2025-05": "May", "2025-06": "Haz",
        "2025-07": "Tem", "2025-08": "Ağu", "2025-09": "Eyl",
        "2025-10": "Eki", "2025-11": "Kas", "2025-12": "Ara",
    }

    for ax, banka in zip(axes, BANKA_SIRASI):
        if banka not in tum_sonuclar:
            continue
        bl    = BANKA_TR[banka]
        renk  = RENKLER[banka]
        sonuc = tum_sonuclar[banka]
        konu_n = sonuc["optimal_k"]

        konu_sut = f"lda_konu_{banka}"
        if konu_sut not in veri.columns:
            continue

        alt = veri[veri["banka_label"] == bl].dropna(
            subset=[konu_sut, "ay_str"])
        alt["konu_int"] = alt[konu_sut].astype(int)

        aylik = alt.groupby(["ay_str", "konu_int"]).size().reset_index(
            name="sayi")
        aylar = sorted(aylik["ay_str"].unique())
        x_pos = range(len(aylar))

        ax.set_facecolor("white")
        konular_d = sonuc["konular"]
        konu_renkler = plt.cm.tab10(np.linspace(0, 0.9, konu_n))

        for k_idx in range(konu_n):
            k_veri = aylik[aylik["konu_int"] == k_idx].set_index("ay_str")
            y_vals = [k_veri.loc[a, "sayi"] if a in k_veri.index else 0
                      for a in aylar]

            konu_adi = f"Konu_{k_idx+1}"
            etiket   = konular_d.get(konu_adi, {}).get(
                "etiket", f"Konu {k_idx+1}")[:22]

            ax.plot(list(x_pos), y_vals,
                    marker="o", ms=4, lw=2,
                    color=konu_renkler[k_idx],
                    label=f"K{k_idx+1}: {etiket}")

        ax.set_xticks(list(x_pos))
        ax.set_xticklabels(
            [AY_TR.get(a, a) for a in aylar],
            fontsize=8, rotation=45)
        ax.set_title(f"{bl}", fontsize=11, fontweight="bold",
                     color=renk)
        ax.set_xlabel("Ay (2025)", fontsize=9)
        ax.set_ylabel("Şikayet Sayısı" if ax == axes[0] else "",
                      fontsize=9)
        ax.legend(fontsize=7, loc="upper right",
                  framealpha=0.85, ncol=1)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.3, ls="--")

    plt.tight_layout()
    plt.savefig("results/figures/20_lda_konu_trend.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 20_lda_konu_trend.png")

    # ─── FİGÜR 21: Konu Kelime Bubble Chart ──────────────────
    # Her konu için top-8 kelimeyi bubble boyutu = olasılık ağırlığı
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        "LDA Konu-Kelime Ağırlık Grafiği\n"
        "Her satır bir konu · Her kabarcık bir anahtar kelime · "
        "Kabarcık boyutu = olasılık ağırlığı",
        fontsize=13, fontweight="bold", color="white", y=1.01
    )

    for ax, banka in zip(axes, BANKA_SIRASI):
        if banka not in tum_sonuclar:
            continue
        bl     = BANKA_TR[banka]
        renk   = RENKLER[banka]
        sonuc  = tum_sonuclar[banka]
        konu_n = sonuc["optimal_k"]

        # Model yükle
        model_path = f"models/lda_{banka.lower()}.model"
        if not os.path.exists(model_path):
            continue
        model_yuk = LdaModel.load(model_path)

        ax.set_facecolor("#0D1117")
        ax.spines[:].set_visible(False)
        ax.tick_params(colors="white", labelsize=8)

        konu_renkler = plt.cm.tab10(np.linspace(0, 0.9, konu_n))
        TOP_N = 8

        for k_idx in range(konu_n):
            konu_adi = f"Konu_{k_idx+1}"
            konular_d = sonuc["konular"]
            etiket = konular_d.get(konu_adi, {}).get(
                "etiket", f"Konu {k_idx+1}")[:20]

            top_kw = model_yuk.show_topic(k_idx, topn=TOP_N)
            kelimeler = [w for w, _ in top_kw]
            agirliklar = [p for _, p in top_kw]
            max_a = max(agirliklar)

            y_val = konu_n - 1 - k_idx  # yukarıdan aşağıya
            for xi, (kw, ag) in enumerate(zip(kelimeler, agirliklar)):
                size = 800 * (ag / max_a) ** 0.6 + 200
                ax.scatter(xi, y_val, s=size,
                           color=konu_renkler[k_idx],
                           alpha=0.75, edgecolors="none")
                ax.text(xi, y_val, kw,
                        ha="center", va="center",
                        fontsize=max(7, int(9 * (ag/max_a)**0.3)),
                        color="white", fontweight="bold",
                        zorder=5)

            # Konu etiketi (sol taraf)
            ax.text(-0.8, y_val,
                    f"K{k_idx+1}\n{etiket[:18]}",
                    ha="right", va="center", fontsize=7.5,
                    color=konu_renkler[k_idx], fontweight="bold")

        ax.set_xlim(-1.2, TOP_N - 0.5)
        ax.set_ylim(-0.8, konu_n - 0.2)
        ax.set_xticks(range(TOP_N))
        ax.set_xticklabels(
            [f"#{i+1}" for i in range(TOP_N)],
            color="gray", fontsize=8)
        ax.set_yticks([])
        ax.set_title(bl, fontsize=12, fontweight="bold",
                     color=renk, pad=10)
        ax.grid(axis="x", alpha=0.1, color="white", ls="--")
        ax.set_xlabel("Kelime Sıralaması (ağırlığa göre)",
                      fontsize=9, color="gray")

    plt.tight_layout(pad=1.5)
    plt.savefig("results/figures/21_lda_konu_kelime_bubble.png",
                dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print("  ✓ 21_lda_konu_kelime_bubble.png")

    # ─── FİGÜR 22: Bankalar Arası Konu Isı Haritası ──────────
    # Hangi konu konuları bankalar arasında benzer/farklı?
    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("#F8F9FA")

    # Her banka için top-6 konu kelimelerini düzleştir
    banka_konu_satirlari = []
    banka_etiketleri = []
    for banka in BANKA_SIRASI:
        if banka not in tum_sonuclar:
            continue
        sonuc = tum_sonuclar[banka]
        konu_n = sonuc["optimal_k"]
        for k_idx in range(konu_n):
            konu_adi = f"Konu_{k_idx+1}"
            etiket = sonuc["konular"].get(konu_adi, {}).get(
                "etiket", f"Konu {k_idx+1}")
            coz = sonuc["konu_cozum"].get("Cozulme_Orani", {}).get(
                k_idx, 0)
            n = sonuc["konu_cozum"].get("Sikayet_Sayisi", {}).get(
                k_idx, 0)
            banka_konu_satirlari.append({
                "Banka":          BANKA_TR[banka],
                "Konu":           f"K{k_idx+1}",
                "Etiket":         etiket[:30],
                "Çözüm %":        round(float(coz), 1) if coz else 0,
                "Şikayet Sayısı": int(n) if n else 0,
            })
            banka_etiketleri.append(
                f"{BANKA_TR[banka][:3]}-K{k_idx+1}")

    ozet_df = pd.DataFrame(banka_konu_satirlari)
    ozet_df.to_csv("results/lda/konu_cozum_analizi.csv",
                   index=False, encoding="utf-8-sig")

    # Pivot: Banka × Konu → Çözüm %
    if not ozet_df.empty:
        ozet_df["Banka_Konu"] = (ozet_df["Banka"].str[:3] + "-"
                                  + ozet_df["Konu"])
        pivot = ozet_df.pivot_table(
            index="Etiket", columns="Banka",
            values="Çözüm %", aggfunc="mean")

        if not pivot.empty:
            ax.set_facecolor("white")
            im = ax.imshow(
                pivot.values,
                cmap="RdYlGn", aspect="auto",
                vmin=20, vmax=80
            )
            plt.colorbar(im, ax=ax, label="Çözülme Oranı (%)",
                         shrink=0.6)
            ax.set_xticks(range(len(pivot.columns)))
            ax.set_xticklabels(pivot.columns, fontsize=11,
                               fontweight="bold")
            ax.set_yticks(range(len(pivot.index)))
            ax.set_yticklabels(pivot.index, fontsize=9)
            for i in range(len(pivot.index)):
                for j in range(len(pivot.columns)):
                    val = pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"%{val:.0f}",
                                ha="center", va="center",
                                fontsize=10, fontweight="bold",
                                color="white" if val < 40 or val > 70
                                else "black")

            ax.set_title(
                "Bankalar Arası Konu × Çözüm Oranı Karşılaştırması\n"
                "Aynı konu kategorisi farklı bankalarda farklı çözüm oranına sahip mi?",
                fontsize=12, fontweight="bold", pad=12
            )
            ax.spines[:].set_visible(False)

    plt.tight_layout()
    plt.savefig("results/figures/22_bankalar_konu_kiyaslama.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 22_bankalar_konu_kiyaslama.png")
"""
    # ==========================================================
    # 5. ÖZET
    # ==========================================================
    print("\n" + "=" * 65)
    print("ADIM 4 TAMAMLANDI!")
    print("=" * 65)

    print("\nOptimal Konu Sayıları ve Coherence:")
    for banka, sonuc in tum_sonuclar.items():
        print(f"  {sonuc['banka_tr']:12s} | "
              f"k={sonuc['optimal_k']}  "
              f"Coherence={sonuc['coherence_max']}  "
              f"Perplexity={sonuc['perplexity']}")

    print(f"""
Üretilen dosyalar:
  data/processed/veri_lda.csv                ← Adım 5 bunu kullanır
  results/lda/lda_sonuclar.json
  results/lda/konu_kelime_tablosu.csv        ← Tez tablosu
  results/lda/konu_cozum_analizi.csv         ← H2 hipotezi verisi
  results/lda/ldavis_*.html                  ← Tarayıcıda açın!
  results/figures/17_lda_coherence.png
  results/figures/18_lda_konu_dagilimi.png
  results/figures/19_lda_konu_cozum_isi.png
  results/figures/20_lda_konu_trend.png
  results/figures/21_lda_konu_kelime_bubble.png
  results/figures/22_bankalar_konu_kiyaslama.png

Sıradaki adım: python adim5_tfidf.py
""")