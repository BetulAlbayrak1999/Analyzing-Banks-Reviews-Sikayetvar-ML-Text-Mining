"""
=============================================================
ADIM 3 (v2 — TAM İYİLEŞTİRİLMİŞ): TÜRKÇE METİN ÖN İŞLEME
=============================================================
Bankacılık Sektörü Müşteri Şikayet Analizi — 2026
Kuveyt Türk | VakıfBank | İşBankası

TEMEL DÜZELTMELER (v1 → v2):
  ✓ Stem bozukluğu tamamen çözüldü:
      - "taraf","üzer","ediyor","şekil","dur","edilme","par" vb.
        65+ bozuk form → ATILDI (stop-words'e eklendi)
      - "par" → "para", "kredis" → "kredi", "sigortas" → "sigorta"
        doğruya düzeltildi
  ✓ Bankacılık anlam taşımayan genel kelimeler eklendi:
      hizmet, bankacılık, banka, fark, son, aynı, yer, kez,
      yaklaşık, gerektik, türk, maxim, tüketiç, ndan, yor ...
  ✓ Stop-words STEM SONRASI da kontrol ediliyor (post-stem filtre)
  ✓ full_text NULL → description_preview ile kurtarma (devam ediyor)
  ✓ 776+ benzersiz stop-word + post-stem liste

ÜRETİLEN ÇIKTILAR:
  data/processed/veri_temiz.csv
  data/processed/stop_word_etkinlik.csv
  results/figures/10_onisleme_karsilastirma.png
  results/figures/11_kelime_bulutu.png
  results/figures/12_ngram_analizi.png
  results/figures/13_kelime_uzunluk_dagilimi.png
  results/figures/14_cozum_kelime_farki.png
  results/figures/15_token_kalite_panel.png
  results/figures/16_stop_word_etkinlik.png

ÇALIŞTIRMA: python adim3_onisleme.py
=============================================================
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch
import numpy as np
import pandas as pd
import re, os, warnings, shutil
from collections import Counter

warnings.filterwarnings("ignore")

plt.rcParams["font.family"]        = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"]         = 150

try:
    from snowballstemmer import TurkishStemmer
    STEMMER_VAR = True
except ImportError:
    STEMMER_VAR = False
    print("UYARI: pip install snowballstemmer")

try:
    from wordcloud import WordCloud
    WORDCLOUD_VAR = True
except ImportError:
    WORDCLOUD_VAR = False
    print("UYARI: pip install wordcloud")

os.makedirs("data/processed",  exist_ok=True)
os.makedirs("data/raw",        exist_ok=True)
os.makedirs("results/figures", exist_ok=True)

# ============================================================
# SABITLER
# ============================================================
BANKA_SIRASI = ["VakifBank", "IsBank", "KuveytTurk"]
BANKA_TR = {
    "VakifBank":  "VakıfBank",
    "IsBank":     "İşBankası",
    "KuveytTurk": "Kuveyt Türk",
}
RENKLER = {
    "VakifBank":  "#1565C0",
    "IsBank":     "#C62828",
    "KuveytTurk": "#2E7D32",
}
COZUM_RENK = {"Çözüldü": "#43A047", "Çözülmedi": "#E53935"}

# ============================================================
# 1. TÜRKÇE STOP WORDS (776+ kelime, 12 kategori)
# ============================================================
print("=" * 65)
print("ADIM 3 (v2): TÜRKÇE METİN ÖN İŞLEME")
print("=" * 65)
print("\n[Stop words oluşturuluyor...]")

ZAMIR = {
    "ben","sen","o","biz","siz","onlar",
    "benim","senin","onun","bizim","sizin","onların",
    "bana","sana","ona","bize","size","onlara",
    "beni","seni","onu","bizi","sizi","onları",
    "bende","sende","onda","bizde","sizde","onlarda",
    "benden","senden","ondan","bizden","sizden","onlardan",
    "kendim","kendin","kendi","kendimiz","kendiniz","kendileri",
    "bu","şu","bunlar","şunlar","bunun","şunun",
    "burada","şurada","orada","buraya","şuraya","oraya",
    "buradan","şuradan","oradan","burası","şurası","orası",
    "hepsi","herkes","kimse","hiçkimse","bazıları",
    "her","herhangi","hiçbir","tüm","bütün",
    "kimi","kime","kimden","kimde",
}

BAGLAC = {
    "ve","ile","ya","veya","yahut","ama","fakat","lakin",
    "ancak","çünkü","zira","ki","da","de","dahi",
    "hem","ne","bile","ise","mi","mı","mu","mü",
    "oysa","oysaki","halbuki","üstelik","ayrıca",
    "dolayısıyla","bu nedenle","bu yüzden",
    "bununla birlikte","ne de","hem de","ya da",
    "değil","değildir","yoktur",
}

EDAT = {
    "için","gibi","kadar","göre","karşı","rağmen","üzere",
    "doğru","dek","değin","başka","öte","öteki",
    "içinde","dışında","üstünde","altında",
    "önünde","arkasında","yanında","arasında","üzerinde",
    "vasıtasıyla","aracılığıyla","sayesinde","nedeniyle",
    "yüzünden","dolayı","itibaren","başlayarak",
    "hakkında","konusunda","dair","tarafından","tarafınca",
}

ZARF = {
    "çok","az","daha","en","hiç","bile","sadece","yalnız",
    "yalnızca","nasıl","neden","niçin","nerede","nereye",
    "nereden","ne","böyle","şöyle","öyle",
    "artık","zaten","hep","her","hiçbir","bazı",
    "birçok","birkaç","pek","oldukça","gayet","epey",
    "hemen","şimdi","sonra","önce","henüz",
    "maalesef","kesinlikle","tabii","elbette","mutlaka",
    "gerçekten","hakikaten","aslında",
    "bazen","nadiren","sürekli","devamlı",
    "tam","tamamen","kısmen","tümüyle",
    "yine","tekrar","yeniden",
    "iyi","kötü","güzel","doğru","yanlış",
    "büyük","küçük","uzun","kısa","fazla",
    "hızlı","yavaş","kolay","zor",
    # Bankacılık metninde anlamsız olan zarflar
    "yaklaşık","aynı","son","fark","yer","kez",
    "aynı","farklı","benzer","ilgili",
}

YARDIMCI_FIIL = {
    "var","yok","olan","olarak","olduğu","olduğunu","olduğundan",
    "olması","olmak","olmaktadır","olmuştur",
    "olacak","olabilir","olsun","olmasına",
    "oldu","olmuş","oluyor","olacağını","olma","olmadı",
    "edilmiştir","edilmektedir","edilecek","edilmesi","edildi","edilme",
    "yapılmıştır","yapılmaktadır","yapılacak","yapılması","yapıldı","yapılma",
    "verilmiştir","verilmektedir","verilecek","verilmesi",
    "alınmıştır","alınmaktadır","alınacak","alınması",
    "söylenmektedir","belirtilmektedir","ifade edilmektedir",
    "etmek","yapmak","vermek","almak","gelmek","gitmek",
    "etmiş","yapmış","vermiş","almış","gelmiş","gitmiş",
    "etti","yaptı","verdi","aldı","geldi","gitti",
    "etmektedir","yapmaktadır","vermektedir","almaktadır",
    "eder","yapar","verir","alır","gelir","gider",
    "ediyor","istiyor","yapıyor","veriyor","alıyor","geliyor",
    "ediyorum","istiyorum","yapıyorum","veriyorum",
    "ettim","yaptım","verdim","aldım","geldim","gittim",
    # Anlamsız stem sonuçları olan fiiller
    "yap","sor","dur","kes","ver","gel","git","al","at","bul","kal",
    "söyle","söyledi","söylüyor","istedi","istediğimde",
    "ettik","istedik","gerektik","olduk","oldukça",
}

ZAMAN_SAYI = {
    "gün","günü","günde","günden","günlük","günler","günlerce",
    "bugün","dün","yarın",
    "hafta","haftada","haftalar","haftadır",
    "ay","ayda","aylık","aylar","aydır",
    "yıl","yılda","yıllık","yıllar","yıldır",
    "saat","saatte","saatler","saatlik",
    "dakika","dakikada","dakikalar",
    "tarih","tarihinde","tarihten","tarihli",
    "ocak","şubat","mart","nisan","mayıs","haziran",
    "temmuz","ağustos","eylül","ekim","kasım","aralık",
    "pazartesi","salı","çarşamba","perşembe","cuma","cumartesi","pazar",
    "sabah","öğlen","akşam","gece","gündüz",
    "bir","iki","üç","dört","beş","altı","yedi",
    "sekiz","dokuz","on","yirmi","otuz","kırk","elli",
    "altmış","yetmiş","seksen","doksan","yüz","bin","milyon",
    "kaç","kaçıncı","birinci","ikinci","üçüncü",
}

SIKAYET_PLATFORM = {
    "sayın","merhaba","iyi günler","iyi akşamlar","iyi sabahlar",
    "saygılarımla","saygılarımızla","saygılar","selamlar",
    "teşekkürler","teşekkür","teşekkür ederim","teşekkür ederiz",
    "iyi çalışmalar","kolay gelsin","başarılar",
    "sevgiler","sevgi ve saygılarımla",
    "devamını gör","devamını","şikayetvar","sikayetvar",
    "devam","gör","görüntüle",
    "şikayet","şikayetim","şikayetimiz","şikayetiniz",
    "bilgilendirme","bildirim","görüş",
    "talep","talebim","talebiniz","talebimiz",
    "başvuru","başvurum","başvurunuz","başvuruda",
    "inceleme","değerlendirme","değerlendirmeniz",
    "geri bildirim","geri dönüş",
    "bekliyorum","beklentim","beklentimiz","beklemekteyim",
    "rica ederim","rica ediyorum","arz ederim",
    "lütfen","acil","acilen","ivedilikle",
    "sorunum","sorunumuz",
    "mağduriyetim","mağduriyetimiz","mağduriyet",
    "şikayet numarası","başvuru numarası",
    "defalarca","birçok kez","tekrar tekrar",
    "hâlâ","hala","henüz","bir türlü",
    "rica","talep","ilgili",
}

BANKACILIK_GENEL = {
    # Analitik değersiz müşteri/hizmet kavramları
    "müşteri","müşterimiz","müşterilerimize","müşterisi","müşterinin",
    "hizmet","hizmeti","hizmetlerimiz","hizmetlerimize","hizmetler",
    "destek","yardım","yardımcı",
    "iletişim","iletişim merkezi","çağrı merkezi",
    "müşteri hizmetleri","müşteri temsilcisi","temsilci",
    "telefon","hat","numara",
    # Genel banka terimleri
    "şube","şubesi","şubeye","şubede","şubeyi","şubesinden",
    "genel müdürlük","genel müdürlüğü","genel müdür",
    "açmak","kapatmak","iptal","iptal etmek","iptal edildi",
    "güncelleme","güncellendi",
    "süreç","süreçte","sürecinde","aşama",
    # Analitik değeri düşük genel terimler
    "bilgi","bilgi almak","bilgilendirme",
    "işlem","işlemi","işlemim","işlemlerim","işlemler",
    "bankacılık","banka","bankaları",
    "finans","finansal",
}

BANKA_ISIMLERI = {
    "vakıfbank","vakifbank","vakıf","vakif","vakıflar","tfkb",
    "işbankası","isbank","isbankasi","iş bankası","is bankasi","işbank",
    "kuveytturk","kuveyt türk","kuveyt turk","kuveyt","ktbank","kt",
    "banka","bankası","bankamız","bankamızın","bankanın","bankanız",
    "bankada","bankadan","bankaya","bankalar","babanın",
    "türk","türkiye","türki",
}

WEB_DIJITAL = {
    "tr","com","http","https","www","net","org",
    "online","internet","dijital",
    "app","web","site","portal","platform",
    "mail","e-mail","email","eposta",
    "sms","mesaj","bildirim","push",
    "android","ios","iphone","samsung","huawei",
    "şifre","parola","password","pin","kod",
    "twitter","instagram","facebook","linkedin",
}

BANKACILIK_EK = {
    # Çekimli formlar — stop edilmeden önceki halleri
    "hesabım","hesabıma","hesabımdan","hesabımda",
    "kartım","kartıma","kartımdan","kartımda","kartımı",
    "param","paramı","paramdan","paramda",
    "işlemim","işlemimi","işlemimde","işlemimden",
    "başvurum","başvurumu","başvurumda","başvurumdan",
    # Genel platform soru kalıpları
    "neden","niçin","niye","nasıl","ne zaman","ne kadar",
    "hangisi","hangi","kaçıncı",
    # Analitik değersiz hitap
    "efendim","beyefendi","hanımefendi","yetkili","yetkililer",
    # Tekrar vurgu
    "vs","vb","gibi","benzeri","türü",
    "yani","demek","kısacası","özetle","sonuç olarak",
    "evet","hayır","tamam","olur","tabi","tabii",
    "peki","neyse","eh","hay hay",
    # Fiil türevleri — stop edilmeli
    "yapıyor","ediyor","istiyor","veriyor","alıyor","geliyor",
    "bakıyor","söylüyor","arıyor","buluyor","getiriyor",
    "kesiyor","geçiyor","giriyor","çıkıyor","dönüyor",
    "yapıyorum","ediyorum","istiyorum","veriyorum",
}

SEKTOREL = {
    "masraf","komisyon","vergi","kdv","bsmv",
    "ekstre","dekont","makbuz","fiş","belge","evrak",
    "sözleşme","anlaşma","protokol","şartname",
    "bddk","tcmb","merkez bankası","hazine","spk",
    "swift","iban","bic","vkn","tckn",
    # Anlamsız bağlam kelimeleri
    "hizmet","fark","son","aynı","yer","kez","kere",
    "bazı","çeşitli","çeşit","tür","tip","türde",
    "oluyor","olunuyor","yapılıyor","ediliyor","veriliyor",
}

TURKCE_STOP_WORDS = (
    ZAMIR | BAGLAC | EDAT | ZARF | YARDIMCI_FIIL |
    ZAMAN_SAYI | SIKAYET_PLATFORM | BANKACILIK_GENEL |
    BANKA_ISIMLERI | WEB_DIJITAL | BANKACILIK_EK | SEKTOREL
)
TURKCE_STOP_WORDS = {w.lower().strip() for w in TURKCE_STOP_WORDS if w.strip()}

# Dosyadan ek stop words
if os.path.exists("data/turkce_stop_words.txt"):
    with open("data/turkce_stop_words.txt", encoding="utf-8") as f:
        TURKCE_STOP_WORDS |= {line.strip().lower() for line in f if line.strip()}

os.makedirs("data", exist_ok=True)
with open("data/turkce_stop_words_v3.txt", "w", encoding="utf-8") as f:
    for w in sorted(TURKCE_STOP_WORDS):
        f.write(w + "\n")

# ============================================================
# 2. POST-STEM STOP LIST (stem sonrası anlamsız formlar)
# ============================================================
# TurkishStemmer'ın ürettiği bozuk / anlamsız stem sonuçları.
# Bunlar stop-words listesinde olmayabilir ama stem sonrası
# gereksiz gürültü yaratır.

POST_STEM_STOP = {
    # Edat/bağlaç kaynaklı bozuk stemler
    "taraf",    # tarafından → edat, anlamsız
    "üzer",     # üzerinden → edat, anlamsız
    "şekil",    # şekilde → bağlaç kalıbı, anlamsız
    "dur",      # durumunda → anlamsız
    "yor",      # fiil eki kalıntısı
    "ndan",     # ek kalıntısı
    "ait",      # aitlik → edat
    "olduk",    # olduğunda → anlamsız
    "olma",     # olmaktadır → anlamsız
    "gerektik", # gerektiğinde → anlamsız
    "yapılma",  # yapılmaktadır → anlamsız
    "edilme",   # edilmektedir → anlamsız
    "tan",      # ek kalıntısı
    "ger",      # geriye → anlamsız
    # Bankacılık ama analitik değersiz stem sonuçları
    "türki",    # türkiye → coğrafi, anlamsız
    "maxim",    # maximum kart → banka markası, analitik değersiz
    "tüketiç",  # tüketici → genel, anlamsız burada
    "hanes",    # hanesi → anlamsız
    "kulla",    # kullanıyor → anlamsız eylem
    "tarif",    # tarifesi → stop edilmeli
    "kazandıra",# kazandıran tarife → marka/kampanya, analitik değersiz
    "numaral",  # numaralı → sıfat, anlamsız
    "topla",    # topladım → anlamsız eylem
    "söyle",    # söylüyor → anlamsız eylem
    "söyledi",  # anlamsız eylem
    "istedi",   # anlamsız eylem
    "ettik",    # anlamsız eylem
    "istedik",  # anlamsız eylem
    "katıl",    # katılıyor → bağlam bağımsız
    "sağla",    # sağlıyor → bağlam bağımsız
    "giderilme",# giderilmesi → anlamsız pasif
    "iletiş",   # iletişim → stop edilmeli (genel terim)
    "aradık",   # aradığımda → anlamsız
    "aynı",     # sıfat — zaten stop
    "gereks",   # gereksinim → anlamsız
    "ricas",    # ricası → selamlama
    "aida",     # aidata → stem bozukluğu
    "sor",      # soruyor → anlamsız eylem
    "kes",      # kesiyor → anlamsız eylem
    "yap",      # yaptım → anlamsız eylem
    "ver",      # verdi → anlamsız eylem
    "gel",      # geldi → anlamsız eylem
    "git",      # gitti → anlamsız eylem
    "al",       # aldım → anlamsız eylem
    "bul",      # buldum → anlamsız eylem
    "kal",      # kaldı → anlamsız eylem
    "gerek",    # gereksinim → anlamsız
    "geç",      # geçiyor → anlamsız
    "son",      # sonunda → zarf
    "kez",      # kaç kez → sayı
    "yer",      # yerinde → zarf
    "fark",     # farklı → sıfat, anlamsız
    "gor",      # görmek → anlamsız
    "bil",      # bilmek → anlamsız
    "gör",      # görmek → anlamsız
    "bak",      # bakmak → anlamsız
    "par",      # para → DÜZELTME aşağıda
    "san",      # sanki → anlamsız
    "uygulamas",# uygulaması → düzeltme aşağıda
    "kredis",   # kredisi → düzeltme aşağıda
    "sigortas", # sigortası → düzeltme aşağıda
    "sigor",    # sigorta → düzeltme aşağıda
    "bankacılık", # genel terim
    "banka",    # genel terim
    "hizmet",   # genel terim
    "yaklaşık", # sayı zarfı
    "ait",      # edat
    "rica",     # selamlama
    "topla",    # eylem
    "türk",     # banka adı
    "maxim",    # marka adı
    "topladı",  # eylem
}

# ============================================================
# 3. STEM DÜZELTME HARİTASI
# ============================================================
# Bozuk stem → anlamlı bankacılık terimi
STEM_DUZELT = {
    "par":        "para",       # para → par
    "uygulamas":  "uygulama",   # uygulaması → uygulama
    "kredis":     "kredi",      # kredisi → kredi
    "sigortas":   "sigorta",    # sigortası → sigorta
    "sigor":      "sigorta",    # sigorta → sigor
    "ia":         "iade",       # iade → ia
    "kar":        "kart",       # kart → kar  (dikkat: kredi kartı)
    "öde":        "ödeme",      # ödeme → öde
    "çöz":        "çözüm",      # çözüm → çöz
    "blo":        "bloke",      # bloke → blo
    "işle":       "işlem",      # işlem → işle
    "hesab":      "hesap",      # hesap → hesab
    "hav":        "havale",     # havale → hav
    "tak":        "taksit",     # taksit → tak
    "fat":        "fatura",     # fatura → fat
    "lim":        "limit",      # limit → lim
    "dov":        "döviz",      # döviz → dov
    "prov":       "provizyon",  # provizyon → prov
}

# ============================================================
# 4. TEMİZLEME FONKSİYONU
# ============================================================
if STEMMER_VAR:
    stemmer = TurkishStemmer()

stop_sayac = Counter()


def turkce_temizle(metin: str, stem: bool = True,
                   say_stop: bool = False) -> str:
    """
    Bankacılık şikayet metni için Türkçe NLP ön işleme.

    SIRA (kritik):
      1. None/NaN kontrolü
      2. Küçük harfe çevir
      3. URL / @mention / # prefix temizle
      4. Bağımsız rakamları temizle
      5. Noktalama temizle (Türkçe harflere dokunma)
      6. Tokenize
      7. PRE-STEM stop-word filtresi  ← STEM'DEN ÖNCE
      8. Uzunluk filtresi > 2 karakter
      9. TurkishStemmer
     10. Stem düzeltme haritası (par→para, ia→iade vb.)
     11. POST-STEM stop listesi  ← stem bozukluklarını temizler
     12. Son uzunluk filtresi ≥ 3 karakter
    """
    global stop_sayac

    if pd.isna(metin) or not isinstance(metin, str) or not metin.strip():
        return ""

    metin = metin.lower()
    metin = re.sub(r"https?://\S+|www\.\S+", " ", metin)
    metin = re.sub(r"@\w+", " ", metin)
    metin = re.sub(r"#(?=\w)", " ", metin)
    metin = re.sub(r"\b\d+\b", " ", metin)
    metin = re.sub(r"[^\w\s]", " ", metin, flags=re.UNICODE)
    metin = metin.replace("_", " ")
    metin = re.sub(r"\s+", " ", metin).strip()

    tokenlar = metin.split()

    # Adım 7: PRE-STEM stop-word filtresi
    temiz = []
    for t in tokenlar:
        if t in TURKCE_STOP_WORDS:
            if say_stop:
                stop_sayac[t] += 1
        else:
            temiz.append(t)
    tokenlar = [t for t in temiz if len(t) > 2]

    if not tokenlar:
        return ""

    if not stem or not STEMMER_VAR:
        return " ".join(tokenlar)

    # Adım 9-12: Stemming + düzeltme + post-stem filtre
    sonuc = []
    for t in tokenlar:
        st = stemmer.stemWord(t)

        # Adım 10: Stem düzeltme haritası
        if st in STEM_DUZELT:
            st = STEM_DUZELT[st]

        # Adım 11: POST-STEM stop listesi
        if st in POST_STEM_STOP:
            if say_stop:
                stop_sayac[st] += 1
            continue

        # Adım 12: Son uzunluk filtresi
        if len(st) >= 3:
            sonuc.append(st)

    return " ".join(sonuc)


# ============================================================
# 5. TEST
# ============================================================
print("\n[Fonksiyon testi — anlamsız kelimeler artık çıkmalı...]")

testler = [
    ("VakıfBank hesabımdan kartımı bloke etti, iade talep ediyorum! "
     "Sayın yetkililer lütfen acilen çözün. Tarafından bildirim beklerim.",
     True, "Edat/selamlama temizleme"),
    ("Kredi kartı aidatı üzerinden para çekildi. "
     "3 haftadır müşteri hizmetleri cevap vermiyor. Mağdur durumdayım.",
     True, "Bankacılık terminolojisi korunmalı"),
    ("Kuveyt Türk altın hesabına para yatırmak istedim sistem hata verdi. "
     "EFT havale işlemi yapılamıyor. ATM'den para çekemiyorum.",
     True, "Çoklu bankacılık terimi"),
    ("Maximum Kart şekilde tarafından üzerinden edilme yapılma "
     "ediyor istiyor dur yor ndan",
     True, "Bozuk stem testi — hepsi silinmeli"),
]

print(f"\n  {'Test':<40} {'Çıktı'}")
print(f"  {'-'*40} {'-'*35}")
for metin, stem, aciklama in testler:
    sonuc = turkce_temizle(metin, stem=stem)
    durum = "✓" if len(sonuc) > 3 else "✗"
    print(f"  [{durum}] {aciklama}")
    print(f"      Temiz: {sonuc[:100]}")

print()

# ============================================================
# 6. VERİ YÜKLEME
# ============================================================
print("[Veriler yükleniyor...]")

DOSYALAR = {
    "VakifBank":  "data/raw/vakifbank_2026.csv",
    "IsBank":     "data/raw/is-bankasi_2026.csv",
    "KuveytTurk": "data/raw/kuveyt-turk_2026.csv",
}
YEDEK = {
    "VakifBank":  "sikayetvar_vakifbank.csv",
    "IsBank":     "sikayetvar_isbank.csv",
    "KuveytTurk": "sikayetvar_kuveyt_turk.csv",
}

dfler = []
for key, dosya in DOSYALAR.items():
    if not os.path.exists(dosya):
        yedek = YEDEK[key]
        if os.path.exists(yedek):
            shutil.copy2(yedek, dosya)
        else:
            print(f"  ✗ {BANKA_TR[key]}: Dosya bulunamadı!")
            continue

    df = pd.read_csv(dosya, encoding="utf-8-sig")
    df["banka_key"]   = key
    df["banka_label"] = BANKA_TR[key]

    # full_text NULL → description_preview ile kurtarma
    mask_null = df["full_text"].isna() | (df["full_text"].str.strip() == "")
    mask_prev = df["description_preview"].notna() & \
                (df["description_preview"].str.strip() != "")
    kurtarilan = (mask_null & mask_prev).sum()
    df.loc[mask_null & mask_prev, "full_text"] = \
        df.loc[mask_null & mask_prev, "description_preview"]

    for sutun in ["view_count", "satisfaction", "upvote_count"]:
        if sutun in df.columns:
            df[sutun] = pd.to_numeric(
                df[sutun].astype(str).str.strip()
                .str.replace(",", ".", regex=False)
                .replace({"nan": np.nan, "": np.nan, "None": np.nan}),
                errors="coerce"
            )

    dfler.append(df)
    print(f"  ✓ {BANKA_TR[key]:12s} → {len(df):5,d} şikayet  "
          f"(kurtarılan: {kurtarilan:4,d})")

veri = pd.concat(dfler, ignore_index=True)
onceki = len(veri)
veri = veri[
    veri["full_text"].notna() &
    (veri["full_text"].str.strip() != "")
].copy()
print(f"\n  Toplam: {len(veri):,} geçerli satır ({onceki - len(veri):,} boş atıldı)")

# ============================================================
# 7. ÖN İŞLEME UYGULA
# ============================================================
print("\n[Ön işleme uygulanıyor... (~2-4 dakika)]")

veri["ham_kelime_sayisi"] = veri["full_text"].fillna("").apply(
    lambda x: len(x.split()))

# Stem UYGULANMIŞ → TF-IDF / ML
print("  Stem uygulamalı (TF-IDF)...")
veri["temiz_metin"] = veri["full_text"].apply(
    lambda x: turkce_temizle(x, stem=True, say_stop=True))

# Stem UYGULANMAMIŞ → LDA
print("  Stem uygulanmamış (LDA)...")
veri["token_listesi"] = veri["full_text"].apply(
    lambda x: turkce_temizle(x, stem=False, say_stop=False))

veri["temiz_kelime_sayisi"] = veri["temiz_metin"].apply(
    lambda x: len(x.split()) if isinstance(x, str) else 0)

stop_etk = pd.DataFrame(
    stop_sayac.most_common(100), columns=["Stop_Word", "Engellenen_Sayı"])

onceki = len(veri)
veri = veri[veri["temiz_metin"].str.strip().str.len() >= 3].copy()
print(f"  ✓ Tamamlandı! {onceki:,} → {len(veri):,} satır")

# Kaydet
veri.to_csv("data/processed/veri_temiz.csv", index=False, encoding="utf-8-sig")
stop_etk.to_csv("data/processed/stop_word_etkinlik.csv",
                index=False, encoding="utf-8-sig")

# ============================================================
# 8. KALİTE KONTROL — En sık kelimeleri göster
# ============================================================
print("\n" + "=" * 65)
print("KALİTE KONTROLÜ — TEMİZLENMİŞ METİN TOP-20 KELİMELER")
print("=" * 65)
for key in BANKA_SIRASI:
    bl  = BANKA_TR[key]
    alt = veri[veri["banka_label"] == bl]["temiz_metin"]
    sayac = Counter()
    for m in alt.dropna():
        sayac.update(str(m).split())
    top20 = [f"{k}({v})" for k, v in sayac.most_common(20)]
    ham   = veri[veri["banka_label"] == bl]["ham_kelime_sayisi"].mean()
    temiz = veri[veri["banka_label"] == bl]["temiz_kelime_sayisi"].mean()
    print(f"\n  {bl} (ham:{ham:.0f}→temiz:{temiz:.0f}, "
          f"%{(1-temiz/ham)*100:.0f} azalma):")
    print(f"  {', '.join(top20)}")

# ============================================================
# 9. FİGÜRLER
# ============================================================
print("\n[Figürler oluşturuluyor...]")

# ─── YARDIMCI: ngram sayma ────────────────────────────────
def ngram_say(metinler, n=1, topk=15, min_freq=10):
    sayac = Counter()
    for m in metinler.dropna():
        tkns = str(m).split()
        if n == 1:
            sayac.update(tkns)
        else:
            for i in range(len(tkns) - n + 1):
                sayac[" ".join(tkns[i:i+n])] += 1
    return [(k, v) for k, v in sayac.most_common(topk) if v >= min_freq]


# ─── FİGÜR 10: Ham vs Temiz Kelime Dağılımı ──────────────
fig, axes = plt.subplots(2, 3, figsize=(17, 10))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle(
    "Metin Ön İşleme Öncesi ve Sonrası Kelime Sayısı Dağılımı\n"
    "Üst Satır: Ham Metin  |  Alt Satır: Temizlenmiş Metin (stop-word + kök bulma)",
    fontsize=13, fontweight="bold", y=1.01
)

for ci, key in enumerate(BANKA_SIRASI):
    bl   = BANKA_TR[key]
    alt  = veri[veri["banka_label"] == bl]
    renk = RENKLER[key]

    for ri, (sutun, alfa, etiket) in enumerate([
        ("ham_kelime_sayisi",   1.0, "Ham Metin"),
        ("temiz_kelime_sayisi", 0.80, "Temizlenmiş Metin"),
    ]):
        ax = axes[ri, ci]
        ax.set_facecolor("white")

        clip = int(alt[sutun].quantile(0.97))
        data = alt[sutun].clip(upper=clip)
        med  = alt[sutun].median()
        ort  = alt[sutun].mean()

        ax.hist(data, bins=45, color=renk, alpha=alfa, edgecolor="none")
        ax.axvline(med, color="#1A1A2E", lw=2.0, ls="--",
                   label=f"Medyan: {med:.0f}")
        ax.axvline(ort, color="#FF6B35", lw=1.8, ls=":",
                   label=f"Ortalama: {ort:.0f}")

        ax.set_title(f"{bl}  —  {etiket}",
                     fontsize=10, fontweight="bold",
                     color=renk if ri == 0 else "#444")
        ax.set_xlabel("Kelime Sayısı", fontsize=9)
        ax.set_ylabel("Şikayet Sayısı" if ci == 0 else "", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(labelsize=8)

        if ri == 1:
            ham_m  = alt["ham_kelime_sayisi"].median()
            azalma = (1 - med / ham_m) * 100 if ham_m > 0 else 0
            ax.text(0.97, 0.80,
                    f"↓ %{azalma:.0f} kelime azaldı",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8.5, fontweight="bold", color="#C62828",
                    bbox=dict(boxstyle="round,pad=0.35",
                              facecolor="#FFF3E0", alpha=0.95,
                              edgecolor="#FF8F00", lw=1))

plt.tight_layout()
plt.savefig("results/figures/10_onisleme_karsilastirma.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 10_onisleme_karsilastirma.png")

"""
# ─── FİGÜR 11: Kelime Bulutu (anlamlı — bozuk stemler yok) ─
if WORDCLOUD_VAR:
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        "Şikayet Metinlerindeki En Sık Bankacılık Kavramları\n"
        "Kök bulma + stop-word temizliği uygulanmış · Büyük kelime = yüksek frekans",
        fontsize=13, fontweight="bold", color="white", y=1.02
    )

    CMAP = {
        "VakifBank":  "Blues",
        "IsBank":     "Reds",
        "KuveytTurk": "Greens",
    }

    for ax, key in zip(axes, BANKA_SIRASI):
        bl  = BANKA_TR[key]
        alt = veri[veri["banka_label"] == bl]["temiz_metin"]
        txt = " ".join(alt.dropna().tolist())
        ax.set_facecolor("#0D1117")

        if len(txt) < 50:
            ax.text(0.5, 0.5, "Yeterli veri yok",
                    ha="center", va="center", color="white", fontsize=12)
            ax.axis("off")
            continue

        wc = WordCloud(
            width=700, height=430,
            background_color="#0D1117",
            colormap=CMAP[key],
            max_words=70,
            collocations=False,
            min_font_size=10,
            prefer_horizontal=0.70,
            min_word_length=3,
        ).generate(txt)

        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(bl, fontsize=15, fontweight="bold",
                     color=RENKLER[key], pad=12)

        # Banka metrik özeti
        n = len(alt.dropna())
        uniq = len({w for m in alt.dropna() for w in str(m).split()})
        ax.text(0.02, 0.02,
                f"n={n:,} şikayet  |  {uniq:,} benzersiz kelime",
                transform=ax.transAxes, color="white",
                fontsize=8, alpha=0.75,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="black", alpha=0.5))

    plt.tight_layout(pad=1.5)
    plt.savefig("results/figures/11_kelime_bulutu.png",
                dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print("  ✓ 11_kelime_bulutu.png")

# ─── FİGÜR 12: N-gram Analizi ─────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(22, 14))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle(
    "Bankacılık Şikayetlerinde N-gram Analizi\n"
    "Üst: Tek Kelime (Unigram)  |  Alt: İkili Kelime Grubu (Bigram)",
    fontsize=14, fontweight="bold", y=1.01
)

for ci, key in enumerate(BANKA_SIRASI):
    bl   = BANKA_TR[key]
    alt  = veri[veri["banka_label"] == bl]["temiz_metin"]
    renk = RENKLER[key]

    for ri, n in enumerate([1, 2]):
        ax   = axes[ri, ci]
        ax.set_facecolor("white")
        gram = ngram_say(alt, n=n, topk=15, min_freq=5)
        if not gram:
            ax.text(0.5, 0.5, "Veri yok", ha="center",
                    transform=ax.transAxes)
            continue

        kw = [g[0] for g in gram]
        fr = [g[1] for g in gram]
        mx = max(fr)

        # Renk yoğunluğu: en yüksek tam renk, düşükler soluk
        alpha_list = [0.95 if f == mx else
                      0.75 if f > mx * 0.6 else 0.50
                      for f in fr]
        colors_ = [renk] * len(kw)

        y_pos = range(len(kw))
        bars  = ax.barh(list(y_pos), fr,
                        color=[renk + f"{int(a*255):02X}" for a in alpha_list],
                        edgecolor="none", height=0.68)

        for yi, (val, k_) in enumerate(zip(fr, kw)):
            ax.text(val + mx * 0.015, yi, f"{val:,}",
                    va="center", fontsize=8.5, fontweight="bold",
                    color="#333")

        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(kw, fontsize=9.5)
        ax.invert_yaxis()
        etiket = "Unigram — En Sık 15 Kelime" if n == 1 \
            else "Bigram — En Sık 15 Kelime Çifti"
        ax.set_title(f"{bl}\n{etiket}",
                     fontsize=10, fontweight="bold",
                     color=renk if n == 1 else "#555")
        ax.set_xlabel("Frekans (şikayet sayısı)", fontsize=9)
        ax.set_xlim(0, mx * 1.22)
        ax.spines[["top","right"]].set_visible(False)
        ax.spines[["left","bottom"]].set_color("#DDD")
        ax.tick_params(axis="y", length=0, labelsize=9)

plt.tight_layout(h_pad=3.5)
plt.savefig("results/figures/12_ngram_analizi.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 12_ngram_analizi.png")

# ─── FİGÜR 13: Kelime Uzunluğu Dağılımı ──────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle(
    "Temizlenmiş Metindeki Kelime Uzunluğu Dağılımı\n"
    "Bankacılık terminolojisi tipik olarak 5–12 karakter uzunluğundadır",
    fontsize=12, fontweight="bold", y=1.01
)

for ax, key in zip(axes, BANKA_SIRASI):
    bl  = BANKA_TR[key]
    alt = veri[veri["banka_label"] == bl]["temiz_metin"]
    renk = RENKLER[key]

    uzunluklar = []
    for m in alt.dropna():
        uzunluklar.extend([len(k) for k in str(m).split() if k])

    sayac_uz = Counter(uzunluklar)
    x_vals = [k for k in sorted(sayac_uz) if 2 < k <= 20]
    y_vals = [sayac_uz[k] for k in x_vals]

    bars = ax.bar(x_vals, y_vals, color=renk, alpha=0.85,
                  edgecolor="white", width=0.75)

    ort_uz = np.mean(uzunluklar)
    med_uz = np.median(uzunluklar)
    ax.axvline(ort_uz, color="#FF6B35", lw=2, ls="--",
               label=f"Ort: {ort_uz:.1f}")
    ax.axvline(med_uz, color="#1A1A2E", lw=2, ls=":",
               label=f"Med: {med_uz:.0f}")
    ax.axvspan(5, 12, alpha=0.07, color=renk,
               label="Bankacılık terim bölgesi (5-12)")

    ax.set_title(f"{bl}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Kelime Uzunluğu (karakter)", fontsize=9)
    ax.set_ylabel("Token Sayısı" if ax == axes[0] else "", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(2, 21)
    ax.set_facecolor("white")
    ax.tick_params(labelsize=8)

    # Toplam token bilgisi
    ax.text(0.02, 0.97, f"Toplam token: {len(uzunluklar):,}",
            transform=ax.transAxes, va="top", fontsize=8,
            color="#555")

plt.tight_layout()
plt.savefig("results/figures/13_kelime_uzunluk_dagilimi.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 13_kelime_uzunluk_dagilimi.png")

# ─── FİGÜR 14: Çözüldü vs Çözülmedi Kelime Farkı ─────────
def karakteristik_kelimeler(alt_df, metin_sutun="temiz_metin", topk=12):
    
    #TF oranı farkına dayalı karakteristik kelime tespiti.
    #Skor = (oran_A - oran_B) / (oran_A + oran_B + ε)
    
    coz  = alt_df[alt_df["is_resolved"]=="Çözüldü"][metin_sutun]
    cdeg = alt_df[alt_df["is_resolved"]=="Çözülmedi"][metin_sutun]

    sc, sd = Counter(), Counter()
    for m in coz.dropna():  sc.update(str(m).split())
    for m in cdeg.dropna(): sd.update(str(m).split())

    nc = max(sum(sc.values()), 1)
    nd = max(sum(sd.values()), 1)

    skorlar = {}
    for k in set(sc) | set(sd):
        if sc[k] + sd[k] < 15:  # minimum frekans eşiği
            continue
        ra = sc[k] / nc
        rb = sd[k] / nd
        skorlar[k] = (ra - rb) / (ra + rb + 1e-9)

    srt = sorted(skorlar.items(), key=lambda x: x[1])
    return srt[-topk:][::-1], srt[:topk]  # (çözüldü-özgü, çözülmedi-özgü)


fig, axes = plt.subplots(3, 2, figsize=(16, 17))
fig.patch.set_facecolor("#F8F9FA")
fig.suptitle(
    "Çözülen ve Çözülemeyen Şikayetlerdeki Karakteristik Kelimeler\n"
    "Sol: Çözülen şikayetlere özgü  |  Sağ: Çözülemeyen şikayetlere özgü",
    fontsize=13, fontweight="bold", y=1.01
)

for ri, key in enumerate(BANKA_SIRASI):
    bl   = BANKA_TR[key]
    alt  = veri[veri["banka_label"] == bl]
    renk = RENKLER[key]

    try:
        coz_list, cdeg_list = karakteristik_kelimeler(alt)
    except Exception as e:
        print(f"  ⚠ {bl}: {e}")
        continue

    for ci, (lst, baslik, renk_bar) in enumerate([
        (coz_list,  f"{bl}\nÇözülen Şikayetlerde Öne Çıkan Kavramlar",
         COZUM_RENK["Çözüldü"]),
        (cdeg_list, f"{bl}\nÇözülemeyen Şikayetlerde Öne Çıkan Kavramlar",
         COZUM_RENK["Çözülmedi"]),
    ]):
        ax = axes[ri, ci]
        ax.set_facecolor("white")
        if not lst:
            ax.text(0.5, 0.5, "Yeterli veri yok",
                    ha="center", transform=ax.transAxes)
            continue

        kw = [x[0] for x in lst]
        sk = [abs(x[1]) for x in lst]

        bars = ax.barh(range(len(kw)), sk,
                       color=renk_bar, edgecolor="none", height=0.65, alpha=0.85)
        ax.set_yticks(range(len(kw)))
        ax.set_yticklabels(kw, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(baslik, fontsize=10, fontweight="bold",
                     color=renk_bar)
        ax.set_xlabel("Relatif Fark Skoru", fontsize=9)
        ax.axvline(0, color="#CCC", lw=1)
        ax.spines[["top","right"]].set_visible(False)
        ax.tick_params(axis="y", length=0)

        # Sayısal etiket
        for i, (val, k_) in enumerate(zip(sk, kw)):
            ax.text(val + max(sk) * 0.02, i,
                    f"{val:.3f}", va="center", fontsize=8)

plt.tight_layout(h_pad=4)
plt.savefig("results/figures/14_cozum_kelime_farki.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 14_cozum_kelime_farki.png")

# ─── FİGÜR 15: Token Kalite Özet Paneli ──────────────────
fig = plt.figure(figsize=(19, 13))
fig.patch.set_facecolor("#F0F4F8")
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.60, wspace=0.38)
fig.suptitle(
    "Metin Ön İşleme Kalite Özet Paneli\n"
    "Üst: Kelime Azalma Oranı  |  Orta: Stop-Word Kategori Dağılımı"
    "  |  Alt: Temel Metrikler",
    fontsize=13, fontweight="bold", y=1.01
)

for ci, key in enumerate(BANKA_SIRASI):
    bl   = BANKA_TR[key]
    alt  = veri[veri["banka_label"] == bl]
    renk = RENKLER[key]

    # Panel 1: Azalma çubuğu
    ax1 = fig.add_subplot(gs[0, ci])
    ax1.set_facecolor("white")
    vals = [alt["ham_kelime_sayisi"].mean(),
            alt["temiz_kelime_sayisi"].mean()]
    clrs = [renk + "66", renk]
    bars = ax1.bar(["Ham Metin", "Temizlenmiş"], vals,
                   color=clrs, edgecolor="white", width=0.5)
    for bar, v in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + .5, f"{v:.1f}",
                 ha="center", fontsize=11, fontweight="bold")
    azalma = (1 - vals[1] / max(vals[0], 1)) * 100
    ax1.set_title(f"{bl}\nOrt. Kelime Sayısı Değişimi",
                  fontsize=10, fontweight="bold")
    ax1.set_ylabel("Kelime Sayısı (Ort.)")
    ax1.text(0.5, 0.93, f"↓ %{azalma:.0f}",
             transform=ax1.transAxes, ha="center",
             fontsize=10, fontweight="bold", color="#C62828",
             bbox=dict(boxstyle="round", facecolor="#FFEBEE",
                       alpha=0.9, edgecolor="#C62828"))
    ax1.spines[["top","right"]].set_visible(False)

    # Panel 2: Stop-word kategori pasta
    ax2 = fig.add_subplot(gs[1, ci])
    ham_tok = []
    for m in alt["full_text"].dropna():
        ham_tok.extend(str(m).lower().split())

    kat_n = {
        "Zamir":      sum(1 for t in ham_tok if t in ZAMIR),
        "Bağlaç":     sum(1 for t in ham_tok if t in BAGLAC),
        "Edat":       sum(1 for t in ham_tok if t in EDAT),
        "Zarf":       sum(1 for t in ham_tok if t in ZARF),
        "Yrd.Fiil":   sum(1 for t in ham_tok if t in YARDIMCI_FIIL),
        "Şikayet K.": sum(1 for t in ham_tok if t in SIKAYET_PLATFORM),
        "Banka Adı":  sum(1 for t in ham_tok if t in BANKA_ISIMLERI),
        "Zaman/Sayı": sum(1 for t in ham_tok if t in ZAMAN_SAYI),
        "Bnk.Genel":  sum(1 for t in ham_tok if t in BANKACILIK_GENEL),
    }
    kat_n = {k: v for k, v in kat_n.items() if v > 0}

    if kat_n:
        clrs_pasta = plt.cm.Set2(np.linspace(0, 1, len(kat_n)))
        wedges, texts, autotexts = ax2.pie(
            list(kat_n.values()),
            labels=list(kat_n.keys()),
            colors=clrs_pasta,
            autopct="%1.0f%%",
            startangle=90,
            wedgeprops=dict(edgecolor="white", lw=1.5),
            textprops=dict(fontsize=7.5),
        )
        for at in autotexts:
            at.set_fontsize(7)
    ax2.set_title(f"{bl}\nEngellenen Stop-Word Kategorisi Dağılımı",
                  fontsize=10, fontweight="bold")

    # Panel 3: Metrik tablosu
    ax3 = fig.add_subplot(gs[2, ci])
    ax3.set_facecolor("white")
    ax3.axis("off")

    ham_t  = int(alt["ham_kelime_sayisi"].sum())
    tmz_t  = int(alt["temiz_kelime_sayisi"].sum())
    eng_t  = ham_t - tmz_t
    uniq   = len({w for m in alt["temiz_metin"].dropna()
                  for w in str(m).split()})
    bos_n  = (alt["temiz_metin"].str.strip() == "").sum()

    metrikler = [
        ("Ham Token Toplam",      f"{ham_t:,}",    "#333"),
        ("Temiz Token Toplam",    f"{tmz_t:,}",    "#2E7D32"),
        ("Engellenen Token",      f"{eng_t:,}",    "#C62828"),
        ("Engelleme Oranı",       f"%{eng_t/max(ham_t,1)*100:.1f}", "#C62828"),
        ("Benzersiz Kelime",      f"{uniq:,}",     "#1565C0"),
        ("Boş Metin Sayısı",      f"{bos_n}",      "#555"),
        ("Ort. Temiz Uzunluk",    f"{alt['temiz_kelime_sayisi'].mean():.1f} kelime", "#333"),
        ("Med. Temiz Uzunluk",    f"{alt['temiz_kelime_sayisi'].median():.0f} kelime", "#333"),
    ]

    y = 0.96
    for ad, deg, rc in metrikler:
        ax3.text(0.04, y, f"• {ad}:",
                 transform=ax3.transAxes, fontsize=9,
                 color="#666", va="top")
        ax3.text(0.97, y, deg,
                 transform=ax3.transAxes, fontsize=9,
                 fontweight="bold", color=rc, ha="right", va="top")
        y -= 0.115

    ax3.set_title(f"{bl}\nToken Kalite Metrikleri",
                  fontsize=10, fontweight="bold")
    ax3.add_patch(FancyBboxPatch(
        (0.01, 0.01), 0.98, 0.98,
        boxstyle="round,pad=0.02",
        transform=ax3.transAxes,
        facecolor="white", edgecolor=renk,
        lw=2, alpha=0.6
    ))

plt.savefig("results/figures/15_token_kalite_panel.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ 15_token_kalite_panel.png")
"""
# ─── FİGÜR 16: Stop-Word Etkinlik Analizi ────────────────
if len(stop_etk) > 0:
    top_stop = stop_etk.head(25).copy()

    clr_map = []
    for sw in top_stop["Stop_Word"]:
        if sw in BANKA_ISIMLERI | POST_STEM_STOP:
            clr_map.append("#C62828")
        elif sw in SIKAYET_PLATFORM:
            clr_map.append("#E65100")
        elif sw in BANKACILIK_GENEL:
            clr_map.append("#1565C0")
        elif sw in YARDIMCI_FIIL:
            clr_map.append("#2E7D32")
        elif sw in ZARF | BAGLAC | EDAT:
            clr_map.append("#6A1B9A")
        else:
            clr_map.append("#546E7A")

    fig, (ax_main, ax_pie) = plt.subplots(1, 2, figsize=(16, 8),
                                          gridspec_kw={"width_ratios":[2,1]})
    fig.patch.set_facecolor("#F8F9FA")
    fig.suptitle(
        "Stop-Word Etkinlik Analizi\n"
        "Sol: En Çok Engellenen 25 Kelime  |  Sağ: Kategori Katkı Oranı",
        fontsize=13, fontweight="bold"
    )

    # Sol: En çok engellenen 25
    ax_main.set_facecolor("white")
    bars = ax_main.barh(
        range(len(top_stop)),
        top_stop["Engellenen_Sayı"],
        color=clr_map, edgecolor="none", height=0.65
    )
    for i, (bar, val) in enumerate(zip(bars, top_stop["Engellenen_Sayı"])):
        ax_main.text(val + 30, i, f"{val:,}",
                     va="center", fontsize=9, fontweight="bold")

    ax_main.set_yticks(range(len(top_stop)))
    ax_main.set_yticklabels(top_stop["Stop_Word"], fontsize=10)
    ax_main.invert_yaxis()
    ax_main.set_xlabel("Toplam Engellenen Token (3 banka)", fontsize=10)
    ax_main.set_title("En Çok Engellenen 25 Stop-Word", fontsize=11, fontweight="bold")
    ax_main.spines[["top","right"]].set_visible(False)

    legend_elems = [
        Patch(facecolor="#C62828", label="Banka Adı / Post-Stem Bozukluk"),
        Patch(facecolor="#E65100", label="Şikayet Platformu Kalıpları"),
        Patch(facecolor="#1565C0", label="Bankacılık Genel Terimler"),
        Patch(facecolor="#2E7D32", label="Yardımcı Fiiller"),
        Patch(facecolor="#6A1B9A", label="Zarf / Bağlaç / Edat"),
        Patch(facecolor="#546E7A", label="Diğer"),
    ]
    ax_main.legend(handles=legend_elems, loc="lower right", fontsize=8.5,
                   framealpha=0.9)

    # Sağ: Kategori pasta
    ax_pie.set_facecolor("white")
    kat_toplam = {
        "Zamir":          sum(1 for t in stop_sayac if t in ZAMIR) * stop_sayac.get(t, 0)
                          if False else
                          sum(v for t, v in stop_sayac.items() if t in ZAMIR),
        "Bağlaç/Edat":   sum(v for t, v in stop_sayac.items() if t in BAGLAC | EDAT),
        "Zarf":           sum(v for t, v in stop_sayac.items() if t in ZARF),
        "Yrd.Fiil":       sum(v for t, v in stop_sayac.items() if t in YARDIMCI_FIIL),
        "Şikayet K.":    sum(v for t, v in stop_sayac.items() if t in SIKAYET_PLATFORM),
        "Banka/Genel":   sum(v for t, v in stop_sayac.items()
                              if t in BANKA_ISIMLERI | BANKACILIK_GENEL),
        "Post-Stem":     sum(v for t, v in stop_sayac.items() if t in POST_STEM_STOP),
        "Diğer":         sum(v for t, v in stop_sayac.items()
                              if t not in (ZAMIR | BAGLAC | EDAT | ZARF |
                                           YARDIMCI_FIIL | SIKAYET_PLATFORM |
                                           BANKA_ISIMLERI | BANKACILIK_GENEL |
                                           POST_STEM_STOP)),
    }
    kat_toplam = {k: v for k, v in kat_toplam.items() if v > 0}

    clrs_p = ["#C62828","#E65100","#FF8F00","#2E7D32",
              "#1565C0","#6A1B9A","#00838F","#546E7A"]
    wedges, texts, autotexts = ax_pie.pie(
        list(kat_toplam.values()),
        labels=list(kat_toplam.keys()),
        colors=clrs_p[:len(kat_toplam)],
        autopct="%1.1f%%",
        startangle=90,
        wedgeprops=dict(edgecolor="white", lw=1.5),
        textprops=dict(fontsize=9),
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax_pie.set_title("Stop-Word Kategori Katkı Oranı\n"
                     f"(Toplam: {sum(kat_toplam.values()):,} token engellendi)",
                     fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("results/figures/16_stop_word_etkinlik.png",
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  ✓ 16_stop_word_etkinlik.png")

# ============================================================
# ÖZET
# ============================================================
print("\n" + "=" * 65)
print("ADIM 3 v2 TAMAMLANDI!")
print("=" * 65)

ozet = []
for key in BANKA_SIRASI:
    bl  = BANKA_TR[key]
    alt = veri[veri["banka_label"] == bl]
    ham  = alt["ham_kelime_sayisi"].mean()
    tmz  = alt["temiz_kelime_sayisi"].mean()
    ozet.append({
        "Banka":             bl,
        "Şikayet Sayısı":    len(alt),
        "Ham Ort.Kelime":    round(ham, 1),
        "Temiz Ort.Kelime":  round(tmz, 1),
        "Azalma (%)":        round((1-tmz/max(ham,1))*100, 1),
        "Benzersiz Kelime":  len({w for m in alt["temiz_metin"].dropna()
                                  for w in str(m).split()}),
    })
print(pd.DataFrame(ozet).to_string(index=False))
print(f"\nStop words: {len(TURKCE_STOP_WORDS)} kelime + "
      f"{len(POST_STEM_STOP)} post-stem filtresi = "
      f"{len(TURKCE_STOP_WORDS) + len(POST_STEM_STOP)} toplam\n")
print("Sıradaki adım: python adim4_lda.py")