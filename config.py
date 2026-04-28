# config.py - Proje Yapılandırması

# Veri dosyaları
VERI_DOSYALARI = {
    "VakifBank": "data/raw/sikayetvar_vakifbank.csv",
    "IsBank": "data/raw/sikayetvar_isbank.csv",
    "KuveytTurk": "data/raw/sikayetvar_kuveyt_turk.csv",
}

# Banka görünen adları (grafik başlıkları için)
BANKA_ADLARI = {
    "VakifBank": "VakıfBank",
    "IsBank": "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}

# Renk paleti (grafiklerde banka renkleri)
BANKA_RENKLERI = {
    "VakifBank": "#1E90FF",    # Mavi
    "IsBank": "#FF6347",       # Kırmızı
    "KuveytTurk": "#32CD32",   # Yeşil
}

# LDA parametreleri
LDA_KONU_SAYISI = 8          # Başlangıç: 8 konu
LDA_PASSES = 15              # Eğitim geçişi
LDA_ALPHA = "auto"

# ML model parametreleri
TEST_BOYUTU = 0.2            # %20 test seti
RANDOM_STATE = 42

# Hedef değişken
HEDEF_DEGISKEN = "is_resolved"   # "Çözüldü" / "Bilinmiyor"

# Stop words dosyası
STOP_WORDS_DOSYA = "data/turkce_stop_words.txt"

# Çıktı klasörleri
CIKTI_MODELLER = "models/"
CIKTI_FIGURLER = "results/figures/"
CIKTI_RAPORLAR = "results/reports/"
CIKTI_LDA = "results/lda/"
