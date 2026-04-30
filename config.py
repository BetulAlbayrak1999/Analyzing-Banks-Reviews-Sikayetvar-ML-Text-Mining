# config.py — Proje Yapılandırması
# Bankacılık Sektörü Müşteri Şikayet Analizi — 2025

# ----------------------------------------------------------
# VERİ DOSYALARI
# ----------------------------------------------------------
VERI_DOSYALARI = {
    "VakifBank":   "data/raw/sikayetvar_vakifbank.csv",
    "IsBank":      "data/raw/sikayetvar_isbank.csv",
    "KuveytTurk":  "data/raw/sikayetvar_kuveyt_turk.csv",
}

# Grafik başlıklarında kullanılacak Türkçe adlar
BANKA_ADLARI = {
    "VakifBank":   "VakıfBank",
    "IsBank":      "İşBankası",
    "KuveytTurk":  "Kuveyt Türk",
}

# Renk paleti
BANKA_RENKLERI = {
    "VakifBank":   "#1565C0",   # Koyu mavi
    "IsBank":      "#B71C1C",   # Koyu kırmızı
    "KuveytTurk":  "#1B5E20",   # Koyu yeşil
}

# ----------------------------------------------------------
# MERKEZİ EĞILIM POLİTİKASI
# ----------------------------------------------------------
# Satisfaction (1-5) dağılımı üç bankada da güçlü biçimde
# çarpık (skewed):  1 → çok yüksek, 2-3 → çok düşük.
# Bu nedenle:
#   ✗ YANLIŞ → ortalama (mean):  aykırı değerlere duyarlı
#   ✓ DOĞRU  → medyan:           çarpık dağılımda güvenilir
#
# Kural: satisfaction içeren HER hesaplamada medyan kullan.
MERKEZI_EGILIM = "median"   # "mean" KULLANMA

# ----------------------------------------------------------
# LDA PARAMETRELERİ
# ----------------------------------------------------------
LDA_KONU_ARALIK  = (3, 8)   # k=3'ten k=7'a kadar aranır
LDA_PASSES       = 20
LDA_ITERATIONS   = 100
LDA_ALPHA        = "auto"
LDA_ETA          = "auto"
MIN_KELIME_FREQ  = 3         # Sözlükte en az 3 belgede geçmeli
MAX_KELIME_ORAN  = 0.85      # Belgelerinin %85'inden fazlasında geçen → çıkar

# ----------------------------------------------------------
# MAKİNE ÖĞRENMESİ PARAMETRELERİ
# ----------------------------------------------------------
TEST_BOYUTU      = 0.20      # %80 eğitim / %20 test
RANDOM_STATE     = 42
CV_FOLD          = 5         # Çapraz doğrulama k-katı
H3_ESIK          = 0.70      # H3 hipotezi: %70 accuracy eşiği

# TF-IDF
TFIDF_MAX_FEATURES = 10_000
TFIDF_NGRAM        = (1, 2)  # Unigram + bigram

# ----------------------------------------------------------
# HEDEF DEĞİŞKEN
# ----------------------------------------------------------
HEDEF_DEGISKEN = "is_resolved"
POZITIF_SINIF  = "Çözüldü"      # Binary: 1
NEGATIF_SINIF  = "Bilinmiyor"   # Binary: 0

# ----------------------------------------------------------
# DOSYA YOLLARI
# ----------------------------------------------------------
STOP_WORDS_DOSYA  = "data/turkce_stop_words.txt"
CIKTI_MODELLER    = "models/"
CIKTI_FIGURLER    = "results/figures/"
CIKTI_RAPORLAR    = "results/reports/"
CIKTI_LDA         = "results/lda/"
