"""
Şikayetvar sayfa limiti ve tarih kontrolü.
python debug_pages.py
"""
import requests
from bs4 import BeautifulSoup
import re
from datetime import date

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/121.0.0.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}

TURKISH_MONTHS = {
    "Ocak": 1, "Şubat": 2, "Mart": 3, "Nisan": 4,
    "Mayıs": 5, "Haziran": 6, "Temmuz": 7, "Ağustos": 8,
    "Eylül": 9, "Ekim": 10, "Kasım": 11, "Aralık": 12,
}

def parse_date(text):
    if not text:
        return None
    m = re.search(r"(\d{1,2})\s+(\w+)\s+(\d{4})", text)
    if m:
        day, month_str, year = int(m.group(1)), m.group(2), int(m.group(3))
        month = TURKISH_MONTHS.get(month_str)
        if month:
            try: return date(year, month, day)
            except: pass
    m = re.search(r"(\d{1,2})\s+(\w+)", text)
    if m:
        day, month_str = int(m.group(1)), m.group(2)
        month = TURKISH_MONTHS.get(month_str)
        if month:
            try:
                today = date.today()
                d = date(today.year, month, day)
                return date(today.year - 1, month, day) if d > today else d
            except: pass
    return None

def get_page_dates(page_num):
    url = f"https://www.sikayetvar.com/turkcell?page={page_num}"
    resp = requests.get(url, headers=HEADERS, timeout=20)
    soup = BeautifulSoup(resp.text, "lxml")
    cards = soup.find_all("article", class_=lambda c: c and "card-v2" in c)
    if not cards:
        return None, None, 0

    dates = []
    for card in cards:
        t = card.find("div", class_=lambda c: c and "js-tooltip" in c and "time" in c)
        if t:
            d = parse_date(t.get_text(strip=True))
            if d:
                dates.append(d)

    if not dates:
        return None, None, len(cards)
    return dates[0], dates[-1], len(cards)

# Test sayfaları: 200, 220, 240, 260, 280, 300
print("Sayfa | İlk Tarih   | Son Tarih   | Kart Sayısı")
print("-" * 55)
for p in [200, 220, 230, 240, 250, 260, 280, 300, 350, 400]:
    first, last, count = get_page_dates(p)
    print(f"  {p:4d} | {str(first):12} | {str(last):12} | {count}")

# Ayrıca sayfa limitini kontrol et
print("\n--- Son sayfalarda ne oluyor? ---")
for p in [990, 995, 1000, 1001, 1005, 1010]:
    first, last, count = get_page_dates(p)
    print(f"  {p:4d} | {str(first):12} | {str(last):12} | {count}")
    