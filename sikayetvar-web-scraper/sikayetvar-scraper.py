"""
=============================================================
Şikayetvar.com Web Scraping Scripti (v6 - Final)
=============================================================
Hedef Operatörler : Turkcell, Türk Telekom, Vodafone
Tarih Aralığı    : 01.11.2025 – 31.01.2026

Düzeltilen sorunlar (v5 → v6):
  1. Çözüm durumu: "solved-badge" class'ı ile doğru tespit
  2. Sayfa döngüsü: Aynı sayfa tekrar gelirse dur (sonsuz döngü fix)
  3. Pagination: has_next_page yerine ID takibi ile daha güvenilir
  4. Skipping fazı: Çok yeni sayfalar hızlı geçilir, log azaltıldı

Kurulum:
    pip install requests beautifulsoup4 pandas lxml

Çalıştırma:
    python sikayetvar_scraper.py
=============================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import re
import os
from datetime import date, datetime

# =============================================================
# 1. AYARLAR
# =============================================================

OPERATORS = {
    #"Kuveyt Türk": "kuveyt-turk",
    #"Ziraat Bankası": "ziraat-bankasi",
    #"VakifBank":     "vakifbank",
    #"Halkbank" : "halkbank", 1120
    #"Yapı Kredi Bankası": "yapi-kredi-bankasi",1384
    "İş Bank": "is-bankasi"
}

DATE_START = date(2025, 1, 1)
DATE_END   = date(2025,  12, 31)

OUTPUT_DIR = "output"

# ── TEST MODU ─────────────────────────────────────────────────
# True  → sadece TEST_LIMIT kadar şikayet çek, hızlı kontrol
# False → tüm tarihi tara (uzun sürer, 2-4 saat)
TEST_MODE  = False
TEST_LIMIT = 10
# ─────────────────────────────────────────────────────────────

DELAY_LIST   = (1.5, 3.0)   # Liste sayfaları arası bekleme (sn)
DELAY_DETAIL = (1.0, 2.0)   # Detay sayfaları arası bekleme (sn)

# =============================================================
# 2. TARİH PARSE
# =============================================================

TURKISH_MONTHS = {
    "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4,
    "Mayıs": 5, "Haziran": 6, "Temmuz": 7, "Ağustos": 8,
    "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12,
}


def parse_date(text):
    """
    '26 Şubat 00:14'      → date(2026, 2, 26)
    '15 Kasım 2025 09:30' → date(2025, 11, 15)
    """
    if not text:
        return None
    text = text.strip()

    # Yıl var: "15 Kasım 2025"
    m = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", text)
    if m:
        day, month_str, year = int(m.group(1)), m.group(2), int(m.group(3))
        month = TURKISH_MONTHS.get(month_str)
        if month:
            try:
                return date(year, month, day)
            except ValueError:
                pass

    # Yıl yok: "26 Şubat 00:14" veya "26 Şubat"
    m = re.search(r"(\d{1,2})\s+(\w+)", text)
    if m:
        day, month_str = int(m.group(1)), m.group(2)
        month = TURKISH_MONTHS.get(month_str)
        if month:
            try:
                today = date.today()
                d = date(today.year, month, day)
                if d > today:
                    d = date(today.year - 1, month, day)
                return d
            except ValueError:
                pass

    return None


# =============================================================
# 3. HTTP İSTEĞİ
# =============================================================

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.sikayetvar.com/",
}


def fetch_page(url, retries=3):
    """URL'yi çeker. Başarısız olursa 3 kez tekrar dener."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "lxml")
            elif resp.status_code == 429:
                wait = 45 + random.uniform(15, 30)
                print(f"    ⚠ Rate limit! {wait:.0f}sn bekleniyor...")
                time.sleep(wait)
            else:
                print(f"    ⚠ HTTP {resp.status_code} (deneme {attempt}/{retries})")
                time.sleep(5)
        except requests.exceptions.Timeout:
            print(f"    ⚠ Zaman aşımı (deneme {attempt}/{retries})")
            time.sleep(5)
        except requests.exceptions.ConnectionError as e:
            print(f"    ⚠ Bağlantı hatası (deneme {attempt}/{retries}): {e}")
            time.sleep(10)
        except Exception as e:
            print(f"    ⚠ Hata: {e}")
            time.sleep(5)
    return None


# =============================================================
# 4. DETAY SAYFASI PARSE
#    Debug çıktısından doğrulanan class'lar:
#      Tam metin   : complaint-detail-description
#      Şirket cvp  : complaint-reply-container
#      Çözüm       : solved-badge  ← v6'da düzeltildi
#      Puan        : rate-num
# =============================================================

def parse_detail_page(url):
    """
    Şikayetin detay sayfasına gidip ek bilgileri çeker.

    Döndürür:
        full_text      : Şikayetin tam metni
        company_reply  : Şirketin cevabı (yoksa "")
        is_resolved    : "Çözüldü" / "Çözülmedi" / "Bilinmiyor"
        satisfaction   : 1.0–5.0 arası puan (yoksa None)
    """
    result = {
        "full_text":     "",
        "company_reply": "",
        "is_resolved":   "Bilinmiyor",
        "satisfaction":  None,
    }

    soup = fetch_page(url)
    if soup is None:
        return result

    # ── Tam şikayet metni ─────────────────────────────────────
    desc = soup.find(class_="complaint-detail-description")
    if desc:
        result["full_text"] = desc.get_text(separator="\n", strip=True)

    # ── Şirket cevabı ─────────────────────────────────────────
    reply = soup.find(class_="complaint-reply-container")
    if reply:
        result["company_reply"] = reply.get_text(separator="\n", strip=True)

    # ── Çözüm durumu ──────────────────────────────────────────
    # Debug çıktısı: <div class=['solved-badge', 'ga-v', 'ga-c']> → "Çözüldü"
    solved = soup.find(class_=lambda c: c and "solved-badge" in c)
    if solved:
        text = solved.get_text(strip=True)
        if "Çözülmedi" in text:
            result["is_resolved"] = "Çözülmedi"
        elif "Çözüldü" in text:
            result["is_resolved"] = "Çözüldü"
        else:
            result["is_resolved"] = "Çözüldü"   # solved-badge varsa çözülmüş demektir

    # ── Memnuniyet puanı ──────────────────────────────────────
    rate = soup.find(class_="rate-num")
    if rate:
        try:
            result["satisfaction"] = float(rate.get_text(strip=True))
        except ValueError:
            pass

    return result


# =============================================================
# 5. LİSTE SAYFASI KART PARSE
# =============================================================

def parse_cards(soup, operator_name):
    """Bir liste sayfasındaki 24 kartı okur."""
    complaints = []
    cards = soup.find_all("article", class_=lambda c: c and "card-v2" in c)

    for card in cards:
        try:
            row = {"operator": operator_name}

            row["id"] = card.get("data-id", "").strip()

            # Başlık + URL
            h2 = card.find("h2", class_="complaint-title")
            if h2:
                a = h2.find("a")
                row["title"] = a.get_text(strip=True) if a else h2.get_text(strip=True)
                href = a.get("href", "") if a else ""
                row["url"] = "https://www.sikayetvar.com" + href if href else ""
            else:
                row["title"] = ""
                row["url"] = ""

            # Kullanıcı
            user = card.find("span", class_="username")
            row["username"] = user.get_text(strip=True) if user else ""

            # Tarih
            time_div = card.find(
                "div",
                class_=lambda c: c and "js-tooltip" in c and "time" in c
            )
            date_text = time_div.get_text(strip=True) if time_div else ""
            row["date_raw"] = date_text
            row["date"] = parse_date(date_text)

            # Görüntülenme
            view = card.find("span", class_=lambda c: c and "js-view-count" in c)
            try:
                row["view_count"] = int(view.get_text(strip=True)) if view else 0
            except ValueError:
                row["view_count"] = 0

            # Upvote
            try:
                row["upvote_count"] = int(card.get("data-upvoter-count", 0))
            except (ValueError, TypeError):
                row["upvote_count"] = 0

            # Kategoriler
            kw_tags = card.find_all("a", class_=lambda c: c and "sv-model-button" in c)
            row["keywords"] = ", ".join(
                kw.get_text(strip=True).lstrip("#").strip() for kw in kw_tags
            )

            # Kısa önizleme
            desc = (
                card.find("p", class_=lambda c: c and "complaint-description" in c) or
                card.find("a", class_=lambda c: c and "complaint-description" in c)
            )
            if desc:
                for span in desc.find_all("span", class_="ellipsis-text"):
                    span.decompose()
                row["description_preview"] = desc.get_text(strip=True)
            else:
                row["description_preview"] = ""

            complaints.append(row)

        except Exception as e:
            print(f"    ⚠ Kart parse hatası: {e}")
            continue

    return complaints


# =============================================================
# 6. ANA SCRAPE FONKSİYONU
# =============================================================

def scrape_operator(operator_name, slug):
    """
    Bir operatörün tüm şikayetlerini çeker.

    Sayfa döngüsü mantığı:
      - Yeni → Eski sıralı (Şikayetvar default)
      - date > DATE_END   → atla (henüz çok yeni)
      - date aralıkta     → topla
      - date < DATE_START → DUR (artık çok eski)

    Döngü koruması:
      - Sayfa içindeki ilk kart ID'si bir öncekiyle aynıysa → site
        aynı sayfayı tekrar döndürüyor demektir → DUR
    """
    mode_str = f"TEST ({TEST_LIMIT} şikayet)" if TEST_MODE else "TAM TARAMA"
    print(f"\n{'─'*60}")
    print(f"  🔍 {operator_name}  [{mode_str}]")
    print(f"  Tarih: {DATE_START} → {DATE_END}")
    print(f"{'─'*60}")

    collected = []
    page = 1
    last_first_id = None   # Döngü tespiti için önceki sayfanın ilk ID'si

    while True:

        # Test modunda limit kontrolü
        if TEST_MODE and len(collected) >= TEST_LIMIT:
            print(f"  [TEST] {TEST_LIMIT} şikayete ulaşıldı, duruyorum.")
            break

        url = (f"https://www.sikayetvar.com/{slug}"
               if page == 1
               else f"https://www.sikayetvar.com/{slug}?page={page}")

        # Skipping fazında her 50 sayfada bir, collecting fazında her sayfada log
        if page <= 3 or page % 50 == 0:
            print(f"  Sayfa {page:4d} | Toplanan: {len(collected):5d}")

        soup = fetch_page(url)
        if soup is None:
            print(f"  ⚠ Sayfa {page} alınamadı, atlıyorum.")
            page += 1
            time.sleep(5)
            continue

        cards = parse_cards(soup, operator_name)

        if not cards:
            print(f"  Sayfa {page} boş. Bitti.")
            break

        # ── Döngü koruması ────────────────────────────────────
        # Şikayetvar sayfa limiti aşılınca son sayfayı tekrar döndürür.
        # İlk kartın ID'si değişmiyorsa aynı sayfadayız → dur.
        current_first_id = cards[0].get("id")
        if current_first_id and current_first_id == last_first_id:
            print(f"  ⚠ Sayfa {page}: Site aynı sayfayı tekrar döndürüyor.")
            print(f"     Şikayetvar'ın sayfa limiti aşıldı. Tarama tamamlandı.")
            break
        last_first_id = current_first_id

        # ── Tarih filtresi ────────────────────────────────────
        stop = False
        page_added = 0

        for c in cards:
            if TEST_MODE and len(collected) >= TEST_LIMIT:
                stop = True
                break

            d = c.get("date")

            if d is None:
                # Tarih okunamadı → ekle (güvenli taraf)
                collected.append(c)
                page_added += 1

            elif d > DATE_END:
                # Çok yeni → atla, bir sonraki karta geç
                pass

            elif DATE_START <= d <= DATE_END:
                # ✅ Hedef aralıkta
                collected.append(c)
                page_added += 1

            else:
                # d < DATE_START → çok eski → tamamen dur
                print(f"\n  🛑 {d} tarihi {DATE_START} sınırının altına düştü.")
                print(f"     Tarama tamamlandı. Toplanan: {len(collected)}")
                stop = True
                break

        if page_added > 0:
            print(f"  Sayfa {page:4d} | +{page_added} eklendi | Toplam: {len(collected)}")

        if stop:
            break

        page += 1
        time.sleep(random.uniform(*DELAY_LIST))

    # ── DETAY SAYFALARINI ÇEK + ANINDA DOSYAYA YAZ ───────────
    if collected:
        print(f"\n  📋 Liste bitti: {len(collected)} şikayet")
        print(f"  🔗 Detay sayfaları çekiliyor, her şikayet anında kaydediliyor...")

        live_path = _get_live_path(operator_name)
        header_written = os.path.exists(live_path)

        for i, complaint in enumerate(collected):
            url = complaint.get("url", "")
            if not url:
                complaint.update({
                    "full_text": "", "company_reply": "",
                    "is_resolved": "Bilinmiyor", "satisfaction": None,
                })
            else:
                title_short = complaint.get("title", "")[:45]
                print(f"  [{i+1:4d}/{len(collected)}] {title_short}")

                detail = parse_detail_page(url)
                complaint.update(detail)

                time.sleep(random.uniform(*DELAY_DETAIL))

            # Her şikayeti anında dosyaya ekle
            _append_row(complaint, live_path, write_header=not header_written)
            header_written = True

    print(f"\n  ✅ {operator_name} tamamlandı: {len(collected)} şikayet")
    return collected


# =============================================================
# 7. ANINDA KAYIT FONKSİYONLARI
# =============================================================

# CSV sütun sırası
COL_ORDER = [
    "operator", "id", "date", "date_raw",
    "username", "title",
    "full_text", "company_reply", "is_resolved", "satisfaction",
    "description_preview", "keywords",
    "view_count", "upvote_count", "url",
]


def _safe_name(operator_name):
    return (operator_name.lower()
            .replace(" ", "_").replace("ü", "u")
            .replace("ö", "o").replace("ı", "i"))


def _get_live_path(operator_name):
    """Her operator icin canli yazilan CSV dosyasinin yolu."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    suffix = "_TEST" if TEST_MODE else ""
    return os.path.join(OUTPUT_DIR, f"sikayetvar_{_safe_name(operator_name)}{suffix}.csv")


def _append_row(complaint, filepath, write_header=False):
    """Tek bir sikayeti CSV dosyasina aninda ekler (append modu)."""
    row = dict(complaint)
    if isinstance(row.get("date"), date):
        row["date"] = row["date"].isoformat()
    ordered = {col: row.get(col, "") for col in COL_ORDER}
    df = pd.DataFrame([ordered])
    df.to_csv(filepath, mode="a", index=False, header=write_header, encoding="utf-8-sig")


def _save_checkpoint(complaints, operator_name):
    """Operator bittikten sonra ozet log yazar."""
    live_path = _get_live_path(operator_name)
    if os.path.exists(live_path):
        print(f"  \U0001f4be Canli dosya: {live_path}  ({len(complaints)} sikayet yazildi)")


# =============================================================
# 8. KAYDETME
# =============================================================

def save_to_csv(all_complaints):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not all_complaints:
        print("\n❌ Kaydedilecek şikayet yok.")
        return None

    df = pd.DataFrame(all_complaints)
    df["date"] = df["date"].apply(lambda d: d.isoformat() if isinstance(d, date) else "")

    col_order = [
        "operator", "id", "date", "date_raw",
        "username", "title",
        "full_text",        # TAM metin
        "company_reply",    # Şirket cevabı
        "is_resolved",      # Çözüm durumu
        "satisfaction",     # Memnuniyet puanı
        "description_preview", "keywords",
        "view_count", "upvote_count", "url",
    ]
    existing = [c for c in col_order if c in df.columns]
    df = df[existing]

    # Tarih filtresi
    df_out = df[
        (df["date"] == "") |
        ((df["date"] >= DATE_START.isoformat()) & (df["date"] <= DATE_END.isoformat()))
    ].copy()

    suffix = "_TEST" if TEST_MODE else ""

    # Birleşik dosya
    #path_all = os.path.join(OUTPUT_DIR, f"sikayetvar_all{suffix}.csv")
    #df_out.to_csv(path_all, index=False, encoding="utf-8-sig")

    # Operatör bazlı
    for op in df_out["operator"].unique():
        safe = (op.lower()
                .replace(" ", "_").replace("ü", "u")
                .replace("ö", "o").replace("ı", "i"))
        path = os.path.join(OUTPUT_DIR, f"sikayetvar_{safe}{suffix}.csv")
        df_out[df_out["operator"] == op].to_csv(path, index=False, encoding="utf-8-sig")

    # Özet
    print(f"\n{'='*60}")
    print(f"  📊 SONUÇ {'(TEST)' if TEST_MODE else ''}")
    print(f"{'='*60}")
    print(f"  Toplam          : {len(df_out)} şikayet")
    for op in df_out["operator"].unique():
        print(f"  {op:15s}  : {len(df_out[df_out['operator']==op]):,}")

    if "is_resolved" in df_out.columns:
        print(f"\n  Çözüm durumu:")
        for s, n in df_out["is_resolved"].value_counts().items():
            print(f"    {s:15s}: {n}")

    if "satisfaction" in df_out.columns:
        scores = df_out["satisfaction"].dropna()
        if not scores.empty:
            print(f"\n  Ort. memnuniyet : {scores.mean():.2f} / 5")

    valid = df_out[df_out["date"] != ""]["date"]
    if not valid.empty:
        print(f"  Tarih aralığı   : {valid.min()} – {valid.max()}")

    print(f"\n  📁 {path_all}")
    print(f"{'='*60}")

    return df_out


# =============================================================
# 9. ÇALIŞTIR
# =============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  🚀 Şikayetvar Scraper (v6)")
    print(f"  Operatörler : {list(OPERATORS.keys())}")
    print(f"  Tarih       : {DATE_START} → {DATE_END}")
    if TEST_MODE:
        print(f"  MOD         : TEST — her operatörden {TEST_LIMIT} şikayet")
        print(f"  Çıktı       : output/sikayetvar_all_TEST.csv")
    else:
        print(f"  MOD         : TAM TARAMA")
        print(f"  Süre        : operatör başına ~1-2 saat")
    print("=" * 60)

    start_time = datetime.now()
    collected = []

    for op_name, slug in OPERATORS.items():
        try:
            results = scrape_operator(op_name, slug)
            collected.extend(results)
        except KeyboardInterrupt:
            print("\n\n⚠ Durduruldu! Veriler kaydediliyor...")
            break
        except Exception as e:
            print(f"\n❌ {op_name} hatası: {e}")
            import traceback; traceback.print_exc()
            print("Sonraki operatöre geçiliyor...")
            continue

        if op_name != list(OPERATORS.keys())[-1]:
            wait = random.uniform(8, 15)
            print(f"\n  ⏳ {wait:.0f}sn bekleniyor...\n")
            time.sleep(wait)

    save_to_csv(collected)

    print(f"\n  ⏱ Toplam süre: {datetime.now() - start_time}")
    print("  ✅ Tamamlandı!")
