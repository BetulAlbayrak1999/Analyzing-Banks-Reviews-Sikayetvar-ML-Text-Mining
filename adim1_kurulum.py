"""
=============================================================
ADIM 1 (İYİLEŞTİRİLMİŞ): ORTAM KURULUMU VE PROJE YAPISI
=============================================================
Bankacılık Sektörü Şikayet Analizi
Kuveyt Türk | VakıfBank | İşBankası — Şikayetvar 2025

İYİLEŞTİRMELER:
  - Kapsamlı Türkçe stop words (bankacılık + platform + gramer)
  - İsbank CSV dosya adı normalizasyonu (i_ş_bank → isbank)
  - Satisfaction ölçümü için medyan politikası belgelendi
  - Veri kalite ön kontrolü eklendi

ÇALIŞTIRMA: python adim1_kurulum.py
=============================================================
"""

import os
import sys
import shutil

# ----------------------------------------------------------
# 1. KLASÖR YAPISI
# ----------------------------------------------------------
PROJE_KLASORLERI = [
    "data/raw",           # Ham CSV dosyaları
    "data/processed",     # Temizlenmiş veriler
    "data/features",      # TF-IDF, embedding vektörleri
    "notebooks",          # Jupyter notebook'lar
    "models",             # Eğitilmiş modeller (.joblib)
    "results/figures",    # Grafikler ve görseller
    "results/reports",    # Sonuç raporları
    "results/lda",        # LDA görselleştirmeleri
    "scripts",            # Yardımcı scriptler
]

print("=" * 65)
print("TEZ PROJESİ KURULUMU BAŞLIYOR")
print("Bankacılık Sektörü Müşteri Şikayet Analizi — 2025")
print("Kuveyt Türk | VakıfBank | İşBankası")
print("=" * 65)

for klasor in PROJE_KLASORLERI:
    os.makedirs(klasor, exist_ok=True)
    print(f"  ✓ {klasor}/ oluşturuldu")

# ----------------------------------------------------------
# 2. CSV DOSYALARINI data/raw/ ALTINA KOPYALA
#    i_ş_bank.csv → isbank.csv (dosya adı normalizasyonu)
# ----------------------------------------------------------
print("\n[CSV dosyaları kopyalanıyor ve normalize ediliyor...]")

KAYNAK_HEDEF = [
    ("sikayetvar_vakifbank.csv",    "data/raw/sikayetvar_vakifbank.csv"),
    ("sikayetvar_i_ş_bank.csv",     "data/raw/sikayetvar_isbank.csv"),    # ← normalize
    ("sikayetvar_kuveyt_turk.csv",  "data/raw/sikayetvar_kuveyt_turk.csv"),
]

for kaynak, hedef in KAYNAK_HEDEF:
    if os.path.exists(kaynak):
        shutil.copy2(kaynak, hedef)
        print(f"  ✓ {kaynak} → {hedef}")
    elif os.path.exists(hedef):
        print(f"  ✓ {hedef} zaten mevcut")
    else:
        print(f"  ✗ BULUNAMADI: {kaynak}  (lütfen proje klasörüne koy)")

# ----------------------------------------------------------
# 3. NLTK VERİLERİ İNDİR
# ----------------------------------------------------------
print("\n[NLTK verileri indiriliyor...]")
try:
    import nltk
    for pkg in ["punkt", "stopwords", "punkt_tab"]:
        nltk.download(pkg, quiet=True)
    print("  ✓ NLTK verileri indirildi")
except ImportError:
    print("  ✗ NLTK yüklü değil → pip install nltk")

# ----------------------------------------------------------
# 4. KAPSAMLI TÜRKÇE STOP WORDS
#    Kategoriler:
#    A) Temel Türkçe gramer (zamirler, bağlaçlar, edatlar)
#    B) Zaman ve sayı ifadeleri
#    C) Genel şikayet platformu kalıpları
#    D) Bankacılık genel terimleri (analitik değer taşımayan)
#    E) Şikayetvar platformuna özgü kalıplar
#    F) Banka isimleri (konu modellemede gürültü oluşturur)
#    G) İngilizce kökenli ama yaygın kullanılan
# ----------------------------------------------------------
print("\n[Türkçe stop words oluşturuluyor...]")

# A) TEMEL TÜRKÇE GRAMER
ZAMIR = {
    "ben", "sen", "o", "biz", "siz", "onlar",
    "benim", "senin", "onun", "bizim", "sizin", "onların",
    "bana", "sana", "ona", "bize", "size", "onlara",
    "beni", "seni", "onu", "bizi", "sizi", "onları",
    "bende", "sende", "onda", "bizde", "sizde", "onlarda",
    "benden", "senden", "ondan", "bizden", "sizden", "onlardan",
    "ben", "kendim", "kendin", "kendi", "kendimiz", "kendiniz", "kendileri",
    "bu", "şu", "bunlar", "şunlar", "bunun", "şunun",
    "burada", "şurada", "orada", "buraya", "şuraya", "oraya",
    "buradan", "şuradan", "oradan", "burası", "şurası", "orası",
}

BAGLAC = {
    "ve", "ile", "ya", "veya", "yahut", "ama", "fakat", "lakin",
    "ancak", "çünkü", "zira", "ki", "da", "de", "dahi", "de",
    "hem", "ne", "bile", "ise", "mi", "mı", "mu", "mü",
    "ya da", "hem de", "ne de", "oysa", "oysaki", "halbuki",
    "üstelik", "ayrıca", "bunun yanı sıra", "bunun yanında",
    "dolayısıyla", "bunun için", "bu nedenle", "bu yüzden",
    "bununla birlikte", "bununla beraber",
}

EDAT = {
    "için", "gibi", "kadar", "göre", "karşı", "rağmen", "üzere",
    "doğru", "dek", "değin", "başka", "öte", "öteki",
    "ile", "içinde", "dışında", "üstünde", "altında",
    "önünde", "arkasında", "yanında", "arasında", "üzerinde",
    "vasıtasıyla", "aracılığıyla", "sayesinde", "nedeniyle",
    "yüzünden", "dolayı", "itibaren", "başlayarak",
}

ZARF = {
    "çok", "az", "daha", "en", "hiç", "bile", "sadece", "yalnız",
    "yalnızca", "nasıl", "neden", "niçin", "nerede", "nereye",
    "nereden", "ne", "böyle", "şöyle", "öyle",
    "artık", "zaten", "hep", "her", "hiçbir", "bazı",
    "birçok", "birkaç", "pek", "oldukça", "gayet", "epey",
    "hemen", "şimdi", "sonra", "önce", "daha", "henüz",
    "maalesef", "ne yazık ki", "maalesef", "kesinlikle",
    "tabii", "tabii ki", "elbette", "mutlaka", "muhakkak",
}

YARDIMCI_FIIL = {
    "var", "yok", "olan", "olarak", "olduğu", "olduğunu",
    "olması", "olmak", "olmaktadır", "olmuştur",
    "edilmiştir", "yapılmıştır", "bulunmaktadır",
    "edilmektedir", "yapılmaktadır", "verilmiştir",
    "olacak", "olacaktır", "edilecek", "yapılacak",
    "olabilir", "edilebilir", "yapılabilir",
    "olsun", "olması", "edilmesi", "yapılması",
    "etmek", "yapmak", "olmak", "vermek", "almak",
    "etmiş", "yapmış", "olmuş", "vermiş", "almış",
    "etti", "yaptı", "oldu", "verdi", "aldı",
    "etmektedir", "yapmaktadır", "vermektedir",
    "etmekte", "yapmakta", "vermekte",
    "denilmektedir", "belirtilmektedir", "ifade edilmektedir",
    "söz konusu", "bahsi geçen",
}

# B) ZAMAN VE SAYI İFADELERİ
ZAMAN_SAYI = {
    "gün", "günü", "günde", "günden", "günlük", "günler",
    "bugun", "bugün", "dun", "dün", "yarın",
    "hafta", "haftada", "haftalar", "haftadır",
    "ay", "ayda", "aylık", "aylar", "aydır",
    "yıl", "yılda", "yıllık", "yıllar", "yıldır",
    "saat", "saatte", "saatler", "saatlik", "saattir",
    "dakika", "dakikada", "dakikalar",
    "tarih", "tarihinde", "tarihten", "tarihli",
    "ocak", "şubat", "mart", "nisan", "mayıs", "haziran",
    "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık",
    "pazartesi", "salı", "çarşamba", "perşembe",
    "cuma", "cumartesi", "pazar",
    "sabah", "öğlen", "akşam", "gece",
    "bir", "iki", "üç", "dört", "beş", "altı", "yedi",
    "sekiz", "dokuz", "on", "yüz", "bin",
}

# C) GENEL ŞİKAYET PLATFORMU KALIPLARI
SIKAYET_PLATFORM = {
    # Selamlama ve kapanış
    "sayın", "merhaba", "iyi günler", "iyi akşamlar",
    "saygılarımla", "saygılarımızla", "teşekkürler", "teşekkür",
    "teşekkür ederim", "teşekkür ederiz", "iyi çalışmalar",
    "selamlar", "kolay gelsin",
    # Platform kalıpları
    "devamını gör", "devamını", "gor", "gör",
    "şikayetvar", "sikayetvar",
    # Başvuru kalıpları
    "bilgi", "bilgilendirme", "bildirim",
    "görüş", "talep", "talebim", "talebiniz",
    "başvuru", "başvurum", "başvurunuz",
    "inceleme", "incelemeniz", "değerlendirme",
    "ilgili", "ilgililere", "konu", "konuda", "konusu",
    "şikayetim", "şikayetimiz", "şikayetiniz",
    "sorunum", "sorunumuz", "sorunum",
    "geri bildirim", "geri dönüş", "geri dönüşünüz",
    "çözüm", "çözüme", "çözüm bekliyorum",
    "bekliyorum", "beklentim", "beklentimiz",
    "rica", "rica ederim", "rica ediyorum",
    "lütfen", "acil",
}

# D) BANKACILIĞA ÖZGÜ AMA ANALİTİK DEĞERİ DÜŞÜK TERİMLER
BANKACILIK_GENEL = {
    # Müşteri kavramları
    "müşteri", "müşterimiz", "müşterilerimize", "müşterisi",
    "müşterinin", "müşterileri",
    # Hizmet kavramları
    "hizmet", "hizmeti", "hizmetlerimiz", "hizmetiniz",
    "işlem", "işlemi", "işlemim", "işlemlerim",
    "işlemler", "işlemleriniz",
    "ürün", "ürünü", "ürünler",
    "destek", "destekleriniz",
    "yardım", "yardımınız",
    # İletişim
    "iletişim", "iletişim merkezi",
    "müşteri hizmetleri", "çağrı merkezi",
    "telefon", "hat", "numara",
    # Genel banka eylemleri
    "açmak", "kapatmak", "iptal", "iptal etmek",
    "şube", "şubesi", "şubeye", "şubede",
    "genel müdürlük", "genel müdürlüğü",
}

# E) ŞİKAYETVAR PLATFORMUNA ÖZGÜ
PLATFORM_OZGU = {
    "çözüldü", "bilinmiyor", "çözülmedi",
    "bekleniyor", "inceleniyor",
    "cevap", "cevabı", "cevabınız",
    "yanıt", "yanıtı", "yanıtınız",
    "memnuniyetsizim", "memnun", "memnuniyetim",
    "şikayet numarası", "başvuru numarası",
    "yorum", "yorumum",
}

# F) BANKA İSİMLERİ (konu modellemede analitik değer taşımaz)
BANKA_ISIMLERI = {
    "vakıfbank", "vakifbank", "vakıf", "vakif",
    "işbankası", "isbank", "isbankasi", "işbankası",
    "iş bankası", "is bankasi",
    "kuveytturk", "kuveyt türk", "kuveyt turk",
    "kuveyt", "türk",
    "banka", "bankası", "bankamız", "bankamızın",
    "bankanın", "bankanız",
}

# G) WEB / İNGİLİZCE
WEB_INGILIZCE = {
    "tr", "com", "http", "https", "www",
    "online", "internet", "dijital", "mobil",
    "app", "uygulama", "web",
}

# Tüm kategorileri birleştir
TURKCE_STOP_WORDS = (
    ZAMIR | BAGLAC | EDAT | ZARF | YARDIMCI_FIIL |
    ZAMAN_SAYI | SIKAYET_PLATFORM | BANKACILIK_GENEL |
    PLATFORM_OZGU | BANKA_ISIMLERI | WEB_INGILIZCE
)

# NLTK Türkçe stop words ile de birleştir
try:
    from nltk.corpus import stopwords
    nltk_tr = set(stopwords.words("turkish"))
    TURKCE_STOP_WORDS = TURKCE_STOP_WORDS | nltk_tr
    print(f"  ✓ NLTK Türkçe stop words eklendi (+{len(nltk_tr)} kelime)")
except Exception:
    pass

# Küçük harfe normalize et ve kaydet
TURKCE_STOP_WORDS = {w.lower().strip() for w in TURKCE_STOP_WORDS if w.strip()}

with open("data/turkce_stop_words.txt", "w", encoding="utf-8") as f:
    for kelime in sorted(TURKCE_STOP_WORDS):
        f.write(kelime + "\n")

print(f"  ✓ {len(TURKCE_STOP_WORDS)} stop word kaydedildi → data/turkce_stop_words.txt")

# Kategori dağılımını raporla
print(f"\n  Stop Words Kategori Raporu:")
print(f"    Zamirler           : {len(ZAMIR)}")
print(f"    Bağlaçlar          : {len(BAGLAC)}")
print(f"    Edatlar            : {len(EDAT)}")
print(f"    Zarflar            : {len(ZARF)}")
print(f"    Yardımcı fiiller   : {len(YARDIMCI_FIIL)}")
print(f"    Zaman/sayı         : {len(ZAMAN_SAYI)}")
print(f"    Şikayet kalıpları  : {len(SIKAYET_PLATFORM)}")
print(f"    Bankacılık genel   : {len(BANKACILIK_GENEL)}")
print(f"    Platform özgü      : {len(PLATFORM_OZGU)}")
print(f"    Banka isimleri     : {len(BANKA_ISIMLERI)}")
print(f"    Web/İngilizce      : {len(WEB_INGILIZCE)}")

# ----------------------------------------------------------
# 5. VERİ KALİTE ÖN KONTROLÜ
# ----------------------------------------------------------
print("\n[Veri kalite ön kontrolü yapılıyor...]")

try:
    import pandas as pd

    DOSYALAR = {
        "VakıfBank":    "data/raw/sikayetvar_vakifbank.csv",
        "İşBankası":    "data/raw/sikayetvar_isbank.csv",
        "Kuveyt Türk":  "data/raw/sikayetvar_kuveyt_turk.csv",
    }

    for banka, dosya in DOSYALAR.items():
        if not os.path.exists(dosya):
            print(f"  ✗ {banka}: Dosya bulunamadı — {dosya}")
            continue

        df = pd.read_csv(dosya, encoding="utf-8-sig")
        toplam      = len(df)
        cozuldu     = (df["is_resolved"] == "Çözüldü").sum()
        sat_null    = df["satisfaction"].isna().sum()
        sat_median  = df["satisfaction"].median()  # ← MEDYAN (çarpık dağılım için doğru)
        sat_skew    = df["satisfaction"].skew()

        print(f"\n  {banka}:")
        print(f"    Toplam şikayet     : {toplam}")
        print(f"    Çözüldü / Bilinmiyor: {cozuldu} / {toplam - cozuldu}")
        print(f"    Çözülme oranı      : %{cozuldu/toplam*100:.1f}")
        print(f"    Satisfaction null  : {sat_null} (%{sat_null/toplam*100:.1f})")
        print(f"    Satisfaction medyan: {sat_median}")
        print(f"    Satisfaction çarpıklık (skew): {sat_skew:.2f}")

        # Çarpıklık uyarısı
        if abs(sat_skew) > 1.0:
            print(f"    ⚠ UYARI: Satisfaction dağılımı çarpık (|skew|={abs(sat_skew):.2f} > 1)")
            print(f"      → Ortalama (mean) yanıltıcı olabilir!")
            print(f"      → Tüm analizlerde MEDYAN kullanılacak (config'de kayıtlı)")

except ImportError:
    print("  pandas yüklü değil → pip install pandas")

# ----------------------------------------------------------
# 6. CONFIG DOSYASI
# ----------------------------------------------------------
print("\n[config.py oluşturuluyor...]")

config_icerik = '''# config.py — Proje Yapılandırması
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
LDA_KONU_ARALIK  = (3, 11)   # k=3'ten k=10'a kadar aranır
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
'''

with open("config.py", "w", encoding="utf-8") as f:
    f.write(config_icerik)
print("  ✓ config.py oluşturuldu")

# ----------------------------------------------------------
# ÖZET
# ----------------------------------------------------------
print("\n" + "=" * 65)
print("ADIM 1 TAMAMLANDI!")
print("=" * 65)
print("""
ÖNEMLİ NOTLAR:
  1. Satisfaction ölçümü → her analizde MEDYAN kullanılacak
     (dağılım çarpık: skew > 1, ortalama yanıltıcı)
  2. İsbank dosya adı normalize edildi:
     i_ş_bank.csv → data/raw/sikayetvar_isbank.csv
  3. Stop words: bankacılık + platform + gramer kategorileri

SIRADAKI ADIM:
  pip install -r requirements.txt
  python adim2_eda.py
""")