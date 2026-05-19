"""
vakifbank_2026_scraper.py
==========================
VakifBank 2026 - tum 15 kolon.

Duzeltmeler:
  1) 403 sorunu: paralel thread KALDIRILDI, sirayla + delay ile cekiliyor
     (paralel = site botu taniyor = 403 = bos veri)
  2) satisfaction: genis selector + metin tarama eklendi
  3) Bos satirlar: title veya url yoksa kart atiliyor (silinmis/gizli sikayet)

Strateji:
  - 3 endpoint x 200 sayfa = tum 2026 kapsami
  - Liste + detay ayni geciste, sirayla
  - Her 10 sayfada otomatik kayit (kesinti korumasi)
  - Kaldigi yerden devam eder (seen_ids)

Kurulum:  pip install requests beautifulsoup4 pandas openpyxl
Calistir: python vakifbank_2026_scraper.py
Cikti:    vakifbank_2026.xlsx  +  vakifbank_2026.csv
Sure:     ~2-3 saat (sirayla cekme, guvenli)
"""

import re, time, random, logging, os, copy, json
from datetime import datetime, date
from dataclasses import dataclass, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Yapilandirma
# ─────────────────────────────────────────────────────────────────────────────

OPERATOR   = "VakifBank"
SLUG       = "vakifbank"
BASE_URL   = "https://www.sikayetvar.com"
DATE_START = date(2026,  1,  1)
DATE_END   = date(2026, 4, 30)

OUTPUT_EXCEL = "vakifbank_2026.xlsx"
OUTPUT_CSV   = "vakifbank_2026.csv"

ENDPOINTS = [
    ("comment", "sortField=comment"),
    ("view",    "sortField=view"),
    ("upvote",  "sortField=upvote"),
]

MAX_PAGES        = 200
PAGE_DELAY_MIN   = 2.0
PAGE_DELAY_MAX   = 3.5
# Duzeltme 1: Paralel KALDIRILDI — sirayla, her detay icin ayri delay
DETAIL_DELAY_MIN = 1.2
DETAIL_DELAY_MAX = 2.2
MAX_RETRIES      = 3
SAVE_EVERY_N     = 10

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "tr-TR,tr;q=0.9",
    "Accept": "text/html,application/xhtml+xml,*/*;q=0.8",
    "Referer": f"https://www.sikayetvar.com/{SLUG}",
}

# ─────────────────────────────────────────────────────────────────────────────
# Loglama
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Veri modeli — 15 kolon
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Complaint:
    operator:            str = OPERATOR
    id:                  str = ""
    date:                str = ""
    date_raw:            str = ""
    username:            str = ""
    title:               str = ""
    full_text:           str = ""
    company_reply:       str = ""
    is_resolved:         str = ""
    satisfaction:        str = ""
    description_preview: str = ""
    keywords:            str = ""
    view_count:          str = ""
    upvote_count:        str = ""
    url:                 str = ""

COLUMNS = [
    "operator", "id", "date", "date_raw", "username", "title",
    "full_text", "company_reply", "is_resolved", "satisfaction",
    "description_preview", "keywords", "view_count", "upvote_count", "url",
]

# ─────────────────────────────────────────────────────────────────────────────
# Tarih parse
# ─────────────────────────────────────────────────────────────────────────────

MONTHS = {
    "Ocak": 1, "Şubat": 2, "Subat": 2, "Mart": 3, "Nisan": 4,
    "Mayıs": 5, "Mayis": 5, "Haziran": 6, "Temmuz": 7,
    "Ağustos": 8, "Agustos": 8, "Eylül": 9, "Eylul": 9,
    "Ekim": 10, "Kasım": 11, "Kasim": 11, "Aralık": 12, "Aralik": 12,
}

def parse_date(raw: str) -> Optional[date]:
    raw = re.sub(r"\s+", " ", raw.strip())
    m = re.match(
        r"^(\d{1,2})\s+(\S+?)(?:\s+(\d{4}))?(?:\s+\d{1,2}:\d{2})?$", raw
    )
    if not m:
        return None
    day  = int(m.group(1))
    mon  = MONTHS.get(m.group(2))
    year = int(m.group(3)) if m.group(3) else datetime.now().year
    if not mon:
        return None
    try:
        return date(year, mon, day)
    except ValueError:
        return None

def in_window(d: Optional[date]) -> bool:
    return d is not None and DATE_START <= d <= DATE_END

# ─────────────────────────────────────────────────────────────────────────────
# HTTP
# ─────────────────────────────────────────────────────────────────────────────

def fetch(session: requests.Session, url: str,
          referer: str = "") -> Optional[BeautifulSoup]:
    hdrs = {}
    if referer:
        hdrs["Referer"] = referer
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = session.get(url, timeout=25, headers=hdrs)
            if r.status_code == 403:
                log.warning(f"  403 — {url} (deneme {attempt}/{MAX_RETRIES})")
                time.sleep(10 * attempt)   # 403'te uzun bekle
                continue
            r.raise_for_status()
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"  Hata deneme {attempt}/{MAX_RETRIES}: {e}")
            time.sleep(5 * attempt)
    return None

# ─────────────────────────────────────────────────────────────────────────────
# Liste sayfasi parse
# Duzeltme 3: title veya url yoksa kart atiliyor
# ─────────────────────────────────────────────────────────────────────────────

def parse_list_page(soup: BeautifulSoup) -> list[Complaint]:
    complaints = []
    n_skipped  = 0

    for card in soup.find_all("article", class_="card-v2"):
        c = Complaint()
        try:
            c.id           = card.get("data-id", "")
            c.upvote_count = card.get("data-upvoter-count", "0")

            a = card.select_one("h2.complaint-title a, h3.complaint-title a")
            if a:
                c.title = a.get_text(strip=True)
                href    = a.get("href", "")
                c.url   = (BASE_URL + href) if href.startswith("/") else href

            uname = card.select_one("span.username")
            if uname:
                c.username = uname.get_text(strip=True)

            td = card.select_one("div.time")
            if td:
                c.date_raw = td.get_text(strip=True)
                d = parse_date(c.date_raw)
                c.date = str(d) if d else ""

            desc = card.select_one(
                "p.complaint-description, a.complaint-description"
            )
            if desc:
                c.description_preview = desc.get_text(strip=True)

            kws = card.select("div.keyword-container a")
            c.keywords = " | ".join(kw.get_text(strip=True) for kw in kws)

            vc = card.select_one("span.js-view-count")
            if vc:
                c.view_count = vc.get_text(strip=True)

            # Talep 1: title/url bos olsa bile kaydet — hic bir sey atma
            # Bos satirlari logla ama dahil et
            if not c.title or not c.url:
                n_skipped += 1
                log.debug(f"  Bos kart (silinmis/gizli): id={c.id}  username={c.username}")

            complaints.append(c)

        except Exception as e:
            log.debug(f"Kart parse hatasi: {e}")

    if n_skipped:
        log.info(f"    ({n_skipped} bos/gizli kart iceriyor — yine de kaydediliyor)")

    return complaints

# ─────────────────────────────────────────────────────────────────────────────
# Detay sayfasi parse — tum 15 kolon
# ─────────────────────────────────────────────────────────────────────────────

SKIP_ANCESTORS = [
    "nav", "menu", "footer", "sidebar", "related",
    "similar", "collection", "suggestion", "breadcrumb",
]

def _first_text(soup: BeautifulSoup, selectors: list[str]) -> str:
    """Verilen selector listesinden ilk dolu metni dondurur."""
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            txt = el.get_text(strip=True)
            if txt:
                return txt
    return ""

def parse_detail(soup: BeautifulSoup, c: Complaint) -> None:
    try:
        # ── 7. full_text ──────────────────────────────────────────────────
        section = soup.select_one("article section")
        if section:
            sec = copy.copy(section)
            for sel in [
                "div.keyword-container", "div.attached-field",
                "div.share-buttons", "div.advertisement",
                "div.similar-complaints", "div.related",
            ]:
                for el in sec.select(sel):
                    el.decompose()
            c.full_text = sec.get_text(separator="\n", strip=True)

        if not c.full_text:
            c.full_text = _first_text(soup, [
                "div.complaint-detail-description",
                "div[class*='complaint-text']",
                "div.js-complaint-detail",
                "div.complaint-content",
            ])

        # ── 8. company_reply ──────────────────────────────────────────────
        # Diagnose: div.company-name ve a.icomoon-reply marka adi iceriyor
        # Gercek cevap metni: icomoon-reply'in SONRAKI sibling elementi
        # veya bir ust parent icindeki p/div metni
        reply_texts = []

        # 1) icomoon-reply linkini bul → parent container → icerik metni
        for reply_link in soup.select("a.icomoon-reply"):
            container = reply_link.find_parent(
                "div", class_=lambda x: x and any(
                    k in " ".join(x) for k in ["complaint-answer","brand","reply","answer"]
                )
            )
            if container:
                ct = copy.copy(container)
                # Marka adi, header, butonlari kaldir
                for rm in ct.find_all(["h2","h3","header","button","time",
                                        "div.company-name","span.company-name-text"]):
                    rm.decompose()
                txt = ct.get_text(separator=" ", strip=True)
                # Marka adini temizle
                txt = re.sub(r"^VakıfBank\s*", "", txt).strip()
                if len(txt) > 30:
                    reply_texts.append(txt)

        # 2) Bilinen selector'lar
        if not reply_texts:
            for sel in [
                "div.brand-reply-content",
                "div.complaint-answer-content",
                "div[class*='brand-answer']",
                "div[class*='reply-content']",
                "div[class*='answer-content']",
                "div[class*='complaint-answer']",
            ]:
                for el in soup.select(sel):
                    el_c = copy.copy(el)
                    for rm in el_c.find_all(["header","h2","h3","time","button"]):
                        rm.decompose()
                    txt = el_c.get_text(separator=" ", strip=True)
                    if len(txt) > 30:
                        reply_texts.append(txt)

        c.company_reply = " ||| ".join(dict.fromkeys(reply_texts))

        # ── 9. is_resolved ────────────────────────────────────────────────
        # Diagnose: "Çözüldü"/"Çözülmedi" metni JS ile render ediliyor
        # Requests ile HTML'de bulunmuyor — alternatif yontemler:
        is_resolved = ""

        # 1) Bilinen badge selector'lar
        is_resolved = _first_text(soup, [
            "span.complaint-status",
            "div.status-badge",
            "span[class*='resolved']",
            "div[class*='solution-badge']",
            "div[class*='complaint-state'] span",
            "span[class*='status-label']",
            "div[class*='is-resolved']",
            "div[class*='solved']",
        ])

        # 2) JSON-LD schema.org verisi
        if not is_resolved:
            for script in soup.find_all("script", type="application/ld+json"):
                try:
                    data = json.loads(script.string or "")
                    if isinstance(data, dict):
                        val = (data.get("resolved") or data.get("isResolved")
                               or data.get("isSolved"))
                        if val is True:
                            is_resolved = "Çözüldü"
                            break
                        elif val is False:
                            is_resolved = "Çözülmedi"
                            break
                except Exception:
                    pass

        # 3) data-resolved / data-solved attribute (article veya main elementinde)
        if not is_resolved:
            for el in soup.find_all(True, attrs={"data-resolved": True}):
                val = el.get("data-resolved", "")
                if val in ("true", "1"):
                    is_resolved = "Çözüldü"
                elif val in ("false", "0"):
                    is_resolved = "Çözülmedi"
                break
            if not is_resolved:
                for el in soup.find_all(True, attrs={"data-is-resolved": True}):
                    val = el.get("data-is-resolved", "")
                    is_resolved = "Çözüldü" if val in ("true","1") else "Çözülmedi"
                    break

        # 4) Tum sayfa metninde genis tarama (JS oncesi render edilen kisimlar)
        if not is_resolved:
            page_text = soup.get_text()
            if "Çözüldü" in page_text:
                is_resolved = "Çözüldü"
            elif "Çözülmedi" in page_text:
                is_resolved = "Çözülmedi"

        # 5) resolved=true endpoint'inden geldiyse (c.url icinde resolved var)
        # Bu bilgi scrape() fonksiyonundan tasinir — bos birak, endpoint'ten doldurulur
        c.is_resolved = is_resolved  # bos kalabilir — endpoint bilgisi dogrulama icin kullanilir

        # ── 10. satisfaction ──────────────────────────────────────────────
        # Diagnose dogrulandı: <div class='rating stars js-tooltip'> text='1'
        # text degeri = yildiz sayisi (1-5), direkt aliyoruz
        sat = ""

        # 1) Ana selector: div.rating.stars — text degeri yildiz sayisi
        rating_div = soup.find(
            "div",
            class_=lambda x: x and "rating" in x and "stars" in x
        )
        if rating_div:
            txt = rating_div.get_text(strip=True)
            # Sadece rakam iceriyorsa al (1-5)
            if txt and re.match(r"^[1-5]$", txt):
                sat = txt + "/5"

        # 2) star-wrapper'lardaki background-size: 100% = dolu yildiz
        if not sat and rating_div:
            filled = 0
            for sw in rating_div.select("div.star-wrapper"):
                style = sw.get("style", "")
                if "100% 100%" in style or "100%" in style.split(":")[1] if ":" in style else False:
                    filled += 1
            if filled > 0:
                sat = f"{filled}/5"

        # 3) Fallback: diger selector'lar
        if not sat:
            sat = _first_text(soup, [
                "div.satisfaction-rate",
                "span[class*='satisfaction']",
                "div[class*='user-satisfaction']",
            ])

        c.satisfaction = sat

        # ── 12. keywords (detay sayfasindan daha tam) ─────────────────────
        kw_els = soup.select("div.keyword-container a")
        if kw_els:
            c.keywords = " | ".join(dict.fromkeys(
                el.get_text(strip=True) for el in kw_els
                if el.get_text(strip=True)
            ))

        # ── 13-14. view + upvote ──────────────────────────────────────────
        vc = soup.select_one("span.js-view-count, span.view-count")
        if vc:
            c.view_count = re.sub(r"[^\d]", "", vc.get_text())

        up = soup.select_one(
            "button.upvote-btn span, span.upvote-count, "
            "button[class*='upvote'] span, div[class*='upvote'] span"
        )
        if up:
            c.upvote_count = re.sub(r"[^\d]", "", up.get_text())

    except Exception as e:
        log.debug(f"Detay parse hatasi ({c.url}): {e}")

# ─────────────────────────────────────────────────────────────────────────────
# Detay cekimi — sirayla (paralel degil)
# Duzeltme 1: ThreadPoolExecutor KALDIRILDI
# ─────────────────────────────────────────────────────────────────────────────

def enrich_sequential(
    session: requests.Session,
    complaints: list[Complaint],
    list_url: str,
) -> list[Complaint]:
    """
    Sirayla detay sayfasi ceker.
    Her istek arasinda random delay — 403 riskini minimize eder.
    """
    total = len(complaints)
    log.info(f"  → {total} sikayet icin detay cekiliyor (sirayla)...")

    for i, comp in enumerate(complaints, 1):
        if not comp.url:
            continue

        soup = fetch(session, comp.url, referer=list_url)
        if soup:
            parse_detail(soup, comp)
        else:
            log.warning(f"    Detay alinamadi [{i}/{total}]: {comp.url}")

        # Her 10 detayda bir ilerleme logu
        if i % 10 == 0:
            log.info(f"    {i}/{total} detay tamamlandi")

        time.sleep(random.uniform(DETAIL_DELAY_MIN, DETAIL_DELAY_MAX))

    return complaints

# ─────────────────────────────────────────────────────────────────────────────
# Kaydetme — duplicate kontrollu, incremental, Excel + CSV
# ─────────────────────────────────────────────────────────────────────────────

def save(complaints: list[Complaint], label: str = "") -> int:
    if not complaints:
        return 0

    new_df = pd.DataFrame([asdict(c) for c in complaints], columns=COLUMNS)

    if os.path.exists(OUTPUT_EXCEL):
        try:
            existing     = pd.read_excel(OUTPUT_EXCEL, dtype=str)
            existing_ids = set(existing["id"].dropna().astype(str))
            new_df = new_df[~new_df["id"].astype(str).isin(existing_ids)]
            if new_df.empty:
                log.info(f"    [Kaydet{label}] Yeni satir yok.")
                return 0
            combined = pd.concat([existing, new_df], ignore_index=True)
        except Exception as e:
            log.warning(f"    Okuma hatasi ({e}), yeniden olusturuluyor.")
            combined = new_df
    else:
        combined = new_df

    combined.sort_values("date", ascending=False, inplace=True)
    combined.reset_index(drop=True, inplace=True)
    combined.to_excel(OUTPUT_EXCEL, index=False, engine="openpyxl")
    combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
    added = len(new_df)
    log.info(
        f"    [Kaydet{label}] +{added} satir → "
        f"toplam {len(combined)} | Excel + CSV"
    )
    return added

# ─────────────────────────────────────────────────────────────────────────────
# Ana scraping
# ─────────────────────────────────────────────────────────────────────────────

def scrape() -> None:
    session     = requests.Session()
    session.headers.update(HEADERS)
    seen_ids: set[str] = set()
    grand_total = 0

    # Kaldigi yerden devam
    if os.path.exists(OUTPUT_EXCEL):
        try:
            ex = pd.read_excel(OUTPUT_EXCEL, dtype=str)
            seen_ids.update(ex["id"].dropna().astype(str))
            log.info(f"Mevcut dosyadan {len(seen_ids)} ID yuklendi (tekrar cekilmez).")
        except Exception:
            pass

    for ep_name, ep_param in ENDPOINTS:
        log.info(f"\n{'='*60}")
        log.info(f"Endpoint: {ep_name}  |  {ep_param}")
        log.info(f"{'='*60}")

        ep_total   = 0
        pending: list[Complaint] = []
        pages_done = 0

        for page_num in range(1, MAX_PAGES + 1):
            list_url = (
                f"{BASE_URL}/{SLUG}?{ep_param}"
                if page_num == 1
                else f"{BASE_URL}/{SLUG}?{ep_param}&page={page_num}"
            )
            log.info(f"  Sayfa {page_num:>3}: {list_url}")

            soup = fetch(session, list_url)
            if not soup:
                pages_done += 1
                continue

            cards = parse_list_page(soup)
            if not cards:
                log.info("  Kart bulunamadi — son sayfa.")
                break

            n_in = n_out = n_dup = 0
            for c in cards:
                d = parse_date(c.date_raw)
                if in_window(d):
                    if c.id not in seen_ids:
                        seen_ids.add(c.id)
                        # resolved=true endpoint'inden geliyorsa is_resolved onceden doldur
                        if ep_name == "resolved" and not c.is_resolved:
                            c.is_resolved = "Çözüldü"
                        pending.append(c)
                        n_in += 1
                        ep_total += 1
                    else:
                        n_dup += 1
                else:
                    n_out += 1

            log.info(
                f"           eklendi={n_in}  disari={n_out}  "
                f"dup={n_dup}  ep_toplam={ep_total}  "
                f"genel={grand_total + ep_total}"
            )

            pages_done += 1

            # Her SAVE_EVERY_N sayfada detay cek + kaydet
            if pages_done % SAVE_EVERY_N == 0 and pending:
                enriched    = enrich_sequential(session, pending, list_url)
                added       = save(enriched, label=f" S{page_num}")
                grand_total += added
                pending.clear()

            time.sleep(random.uniform(PAGE_DELAY_MIN, PAGE_DELAY_MAX))

        # Endpoint bitti — kalanlar
        if pending:
            log.info(f"  [{ep_name}] kalan {len(pending)} sikayet isleniyor...")
            enriched    = enrich_sequential(session, pending, list_url)
            added       = save(enriched, label=f" {ep_name}-son")
            grand_total += added
            pending.clear()

        log.info(f"  [{ep_name}] bitti: {ep_total} yeni sikayet.")

    session.close()

    # ── Ozet raporu ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info(f"TAMAMLANDI — toplam {grand_total} yeni sikayet eklendi.")

    if os.path.exists(OUTPUT_EXCEL):
        df = pd.read_excel(OUTPUT_EXCEL, dtype=str)
        df["ym"] = df["date"].str[:7]

        log.info("\n--- Aylik Ozet ---")
        for ym, grp in df.groupby("ym"):
            log.info(f"  {ym}: {len(grp):>5} sikayet") 

        missing = {f"2026-{m:02d}" for m in range(1, 13)} - set(df["ym"].unique())
        if missing:
            log.warning(f"  EKSIK AYLAR: {sorted(missing)}")
        else:
            log.info("  Tum 2026 aylari kapsandi!")

        log.info("\n--- Kolon Doluluk Orani ---")
        for col in COLUMNS:
            if col in df.columns:
                filled = df[col].replace("", pd.NA).notna().sum()
                pct    = filled / len(df) * 100
                flag   = "" if pct > 80 else "  ← DUSUK"
                log.info(f"  {col:<25}: %{pct:>5.1f}{flag}")

        log.info(f"\n  TOPLAM: {len(df)}")
        log.info(f"  Cikti : {OUTPUT_EXCEL}  +  {OUTPUT_CSV}")


# ─────────────────────────────────────────────────────────────────────────────
# Giris
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("VakifBank 2026 Scraper — 15 Kolon")
    log.info(f"Tarih  : {DATE_START}  →  {DATE_END}")
    log.info(f"Cikti  : {OUTPUT_EXCEL}  +  {OUTPUT_CSV}")
    log.info(f"Kayit  : Her {SAVE_EVERY_N} sayfada bir (Excel + CSV)")
    log.info(f"Not    : Sirayla cekme — 403 riski minimize edildi")
    log.info("=" * 60)
    scrape()