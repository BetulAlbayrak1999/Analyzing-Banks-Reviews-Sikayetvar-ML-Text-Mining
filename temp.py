import requests
from bs4 import BeautifulSoup
import time
import pandas as pd
from datetime import datetime

BASE_URL = "https://www.sikayetvar.com/vakifbank"

HEADERS = {
    "User-Agent": "Mozilla/5.0"
}

def parse_date(text):
    months = {
        "Ocak":1,"Şubat":2,"Mart":3,"Nisan":4,
        "Mayıs":5,"Haziran":6,"Temmuz":7,"Ağustos":8,
        "Eylül":9,"Ekim":10,"Kasım":11,"Aralık":12
    }

    parts = text.split()

    if len(parts) >= 3:
        day = int(parts[0])
        month = months.get(parts[1])
        year = int(parts[2])
        return datetime(year, month, day)

    return None


def scrape():
    page = 1
    all_data = []

    seen_ids = set()

    while True:
        url = f"{BASE_URL}?page={page}"

        print(f"Page {page}")

        r = requests.get(url, headers=HEADERS)
        soup = BeautifulSoup(r.text, "lxml")

        cards = soup.find_all("article")

        if not cards:
            break

        stop_flag = False

        for c in cards:
            try:
                cid = c.get("data-id")

                if cid in seen_ids:
                    continue

                seen_ids.add(cid)

                title = c.find("h2").get_text(strip=True)

                date_text = c.find("div", class_="js-tooltip").get_text(strip=True)
                d = parse_date(date_text)

                if d is None:
                    continue

                if d.year == 2025:
                    all_data.append({
                        "id": cid,
                        "title": title,
                        "date": d
                    })

                elif d.year < 2025:
                    stop_flag = True
                    break

            except:
                continue

        if stop_flag:
            print("Reached 2024 → stopping")
            break

        page += 1
        time.sleep(2)

    return pd.DataFrame(all_data)


df = scrape()
df.to_csv("vakif_2025.csv", index=False)

print("DONE:", len(df))