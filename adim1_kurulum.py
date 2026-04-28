"""
=============================================================
ADIM 1: ORTAM KURULUMU VE PROJE YAPISI OLUŞTURMA
=============================================================
Bu scripti VS Code terminalinde bir kez çalıştır.
Tüm klasörleri ve temel yapıyı oluşturur.

ÇALIŞTIRMA: python adim1_kurulum.py
=============================================================
"""

import os
import sys
import subprocess

# ----------------------------------------------------------
# 1. KLASÖR YAPISI
# ----------------------------------------------------------
PROJE_KLASORLERI = [
    "data/raw",          # Ham CSV dosyaları buraya koy
    "data/processed",    # Temizlenmiş veriler
    "data/features",     # TF-IDF, embedding vektörleri
    "notebooks",         # Jupyter notebook'lar
    "models",            # Eğitilmiş modeller (.joblib)
    "results/figures",   # Grafikler ve görseller
    "results/reports",   # Sonuç raporları
    "results/lda",       # LDA görselleştirmeleri
    "scripts",           # Yardımcı scriptler
]

print("=" * 60)
print("TEZ PROJESİ KURULUMU BAŞLIYOR")
print("Bankacılık Sektörü Şikayet Analizi")
print("=" * 60)

# Klasörleri oluştur
for klasor in PROJE_KLASORLERI:
    os.makedirs(klasor, exist_ok=True)
    print(f"  ✓ {klasor}/ oluşturuldu")

# ----------------------------------------------------------
# 2. NLTK VERİLERİ İNDİR
# ----------------------------------------------------------
print("\n[NLTK verileri indiriliyor...]")
try:
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("  ✓ NLTK verileri indirildi")
except ImportError:
    print("  ✗ NLTK yüklü değil. Önce: pip install nltk")

# ----------------------------------------------------------
# 3. TÜRKÇE STOP WORDS DOSYASI OLUŞTUR
# ----------------------------------------------------------
print("\n[Türkçe stop words oluşturuluyor...]")

TURKCE_STOP_WORDS = [
    # Zamirler
    "ben", "sen", "o", "biz", "siz", "onlar",
    "benim", "senin", "onun", "bizim", "sizin", "onların",
    "bana", "sana", "ona", "bize", "size", "onlara",
    "beni", "seni", "onu", "bizi", "sizi", "onları",
    "bende", "sende", "onda", "bizde", "sizde", "onlarda",
    "benden", "senden", "ondan", "bizden", "sizden", "onlardan",
    # Bağlaçlar
    "ve", "ile", "ya", "veya", "yahut", "ama", "fakat", "lakin",
    "ancak", "çünkü", "zira", "ki", "da", "de", "dahi",
    "hem", "ne", "bile", "ise", "de", "mi", "mı", "mu", "mü",
    # Edatlar
    "için", "gibi", "kadar", "göre", "karşı", "rağmen", "üzere",
    "doğru", "dek", "değin", "başka", "başkası", "öte",
    # Zarflar
    "çok", "az", "daha", "en", "hiç", "bile", "sadece", "yalnız",
    "yalnızca", "nasıl", "neden", "niçin", "nerede", "nereye",
    "nereden", "ne", "bu", "şu", "o", "böyle", "şöyle", "öyle",
    "burada", "şurada", "orada", "buraya", "şuraya", "oraya",
    "buradan", "şuradan", "oradan",
    # Fiiller (yardımcı)
    "var", "yok", "olan", "olarak", "olduğu", "olduğunu",
    "olması", "olmak", "olmaktadır", "olmaktadır", "olmuştur",
    "edilmiştir", "yapılmıştır", "bulunmaktadır", "bulunmaktadır",
    "edilmektedir", "yapılmaktadır", "verilmiştir",
    # Sık kullanılan kelimeler (şikayet metinlerinde)
    "sayın", "müşterimiz", "saygılarımızla", "iletişim",
    "bilgi", "bilgilendirme", "görüş", "talep", "talebiniz",
    "başvuru", "başvurunuz", "inceleme", "değerlendirme",
    "ilgili", "ilgililere", "konusu", "konuda", "konu",
    "gün", "tarih", "tarihinde", "saat", "süre",
    "devamını", "gör", "devamını gör",
    # Banka genel terimleri (stopword olarak ekliyoruz)
    "vakıfbank", "işbankası", "kuveytturk", "kuveyt", "türk",
    "banka", "bankası", "bankamız", "bankamızın", "bankanın",
    "müşteri", "müşterimiz", "müşterilerimize",
]

# Kaydet
with open("data/turkce_stop_words.txt", "w", encoding="utf-8") as f:
    for kelime in sorted(set(TURKCE_STOP_WORDS)):
        f.write(kelime + "\n")

print(f"  ✓ {len(TURKCE_STOP_WORDS)} stop word kaydedildi → data/turkce_stop_words.txt")

# ----------------------------------------------------------
# 4. VERİ DOSYALARINI KONTROL ET
# ----------------------------------------------------------
print("\n[Veri dosyaları kontrol ediliyor...]")

BEKLENEN_DOSYALAR = [
    "data/raw/sikayetvar_vakifbank.csv",
    "data/raw/sikayetvar_isbank.csv",        # isim uyarısı aşağıda
    "data/raw/sikayetvar_kuveyt_turk.csv",
]

# ----------------------------------------------------------
# 5. CONFIG DOSYASI OLUŞTUR
# ----------------------------------------------------------
config_icerik = '''# config.py - Proje Yapılandırması

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
'''

with open("config.py", "w", encoding="utf-8") as f:
    f.write(config_icerik)
print("\n  ✓ config.py oluşturuldu")

# ----------------------------------------------------------
# ÖZET
# ----------------------------------------------------------
print("\n" + "=" * 60)
print("KURULUM TAMAMLANDI!")
print("=" * 60)
print("""
SIRADAKI ADIMLAR:
1. CSV dosyalarını data/raw/ klasörüne kopyala
2. İsim düzenlemesi: i_ş_bank.csv → isbank.csv
3. Şu komutu çalıştır:
   pip install -r requirements.txt
4. Ardından: python adim2_eda.py
""")
