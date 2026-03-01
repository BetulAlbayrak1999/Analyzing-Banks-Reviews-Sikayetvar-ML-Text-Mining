"""
DETAY SAYFASI DEBUG SCRIPTİ
Bir şikayetin detay sayfasından tam metin, şirket cevabı,
çözüm durumu ve memnuniyet puanını nerede bulacağımızı görmek için.

Çalıştır:
    python debug_detail.py

Çıktıyı teze yapıştır.
"""

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9",
}

# ─── ADIM 1: Liste sayfasından gerçek bir şikayet URL'i al ───────────────────
print("Liste sayfasından şikayet URL'leri alınıyor...")
list_resp = requests.get("https://www.sikayetvar.com/turkcell", headers=HEADERS, timeout=20)
list_soup = BeautifulSoup(list_resp.text, "lxml")

cards = list_soup.find_all("article", class_=lambda c: c and "card-v2" in c)
print(f"Bulunan kart: {len(cards)}")

# İlk 3 kartın URL'sini al
urls = []
for card in cards[:3]:
    h2 = card.find("h2", class_="complaint-title")
    if h2:
        a = h2.find("a")
        if a and a.get("href"):
            full_url = "https://www.sikayetvar.com" + a["href"]
            urls.append(full_url)

print(f"\nTest edilecek URL'ler:")
for u in urls:
    print(f"  {u}")

# ─── ADIM 2: İlk detay sayfasını incele ──────────────────────────────────────
if not urls:
    print("URL bulunamadı!")
    exit()

print(f"\n{'='*60}")
print(f"DETAY SAYFASI ANALİZİ: {urls[0]}")
print(f"{'='*60}")

detail_resp = requests.get(urls[0], headers=HEADERS, timeout=20)
detail_soup = BeautifulSoup(detail_resp.text, "lxml")

# ── Tam şikayet metni ─────────────────────────────────────────────────────────
print("\n--- TAM ŞİKAYET METNİ ---")
# Olası class'lar
for cls in ["complaint-detail", "complaint-text", "js-complaint-detail",
            "complaint-content", "description", "complaint-body"]:
    el = detail_soup.find(class_=cls)
    if el:
        print(f"  class='{cls}' BULUNDU:")
        print(f"  '{el.get_text(strip=True)[:200]}...'")
        break
else:
    # Bulunamazsa section ve article içeriğine bak
    print("  Bilinen class bulunamadı. section/article içerikleri:")
    for tag in detail_soup.find_all(["section", "article"])[:5]:
        text = tag.get_text(strip=True)
        if len(text) > 100:
            print(f"  <{tag.name} class={tag.get('class')}> → '{text[:150]}...'")

# ── Şirket cevabı ─────────────────────────────────────────────────────────────
print("\n--- ŞİRKET CEVABI ---")
for cls in ["brand-answer", "answer", "company-answer", "brand-response",
            "complaint-answer", "firm-answer"]:
    el = detail_soup.find(class_=cls)
    if el:
        print(f"  class='{cls}' BULUNDU:")
        print(f"  '{el.get_text(strip=True)[:200]}...'")
        break
else:
    print("  Bilinen class bulunamadı.")
    # Cevap genelde "Marka Cevabı" veya firma adıyla başlar
    for tag in detail_soup.find_all(["div", "section"]):
        text = tag.get_text(strip=True)
        if "cevap" in text.lower() or "yanıt" in text.lower():
            cls = tag.get("class", [])
            print(f"  'cevap/yanıt' içeren tag: <{tag.name} class={cls}>")
            print(f"  '{text[:150]}...'")
            break

# ── Çözüm durumu ──────────────────────────────────────────────────────────────
print("\n--- ÇÖZÜM DURUMU ---")
for cls in ["resolved", "complaint-status", "status", "badge",
            "solution-status", "is-resolved"]:
    el = detail_soup.find(class_=cls)
    if el:
        print(f"  class='{cls}' BULUNDU: '{el.get_text(strip=True)}'")
        break
else:
    # "Çözüldü" veya "Çözülmedi" metni ara
    for tag in detail_soup.find_all(["span", "div", "p"]):
        text = tag.get_text(strip=True)
        if text in ["Çözüldü", "Çözülmedi", "Beklemede", "Cevaplandı"]:
            print(f"  '{text}' metni bulundu → <{tag.name} class={tag.get('class')}>")
            break
    else:
        print("  Çözüm durumu bulunamadı.")

# ── Memnuniyet puanı ──────────────────────────────────────────────────────────
print("\n--- MEMNUNİYET PUANI ---")
# itemprop, rate, score gibi attribute'lar dene
for attr in [("itemprop", "ratingValue"), ("class", "rate-num"),
             ("class", "rating"), ("class", "score")]:
    el = detail_soup.find(attrs={attr[0]: attr[1]})
    if el:
        print(f"  {attr[0]}='{attr[1]}' BULUNDU: '{el.get_text(strip=True)}'")
        break
else:
    print("  Bilinen attribute bulunamadı.")

# ── Tüm önemli class'ları listele ─────────────────────────────────────────────
print("\n--- SAYFADAKI TÜM ÖNEMLİ CLASS'LAR (ipucu için) ---")
seen = set()
for tag in detail_soup.find_all(True):
    classes = tag.get("class", [])
    for cls in classes:
        if cls not in seen and any(kw in cls for kw in
            ["complaint", "answer", "brand", "resolve", "rate",
             "status", "score", "detail", "content", "text", "firm"]):
            seen.add(cls)
            text = tag.get_text(strip=True)[:60]
            print(f"  .{cls} → '{text}'")
            