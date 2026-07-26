# Analyzing Banks Reviews — Şikayetvar NLP & Machine Learning

> **Tez çalışması / Thesis project** — Marmara University  
> Comparative text mining and machine learning analysis of online banking complaints from [Şikayetvar](https://www.sikayetvar.com).

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Analysis-2026%20data-informational.svg)](#dataset)

---

## Thesis title

**TR:** Bankacılık Sektöründe Çevrimiçi Müşteri Şikayetlerinin Metin Madenciliği ve Makine Öğrenmesi Teknikleri ile İncelenmesi: Kuveyt Türk, VakıfBank ve İşBankası Üzerine Karşılaştırmalı Bir Araştırma

**EN:** Analyzing Online Customer Complaints in the Banking Sector with Text Mining and Machine Learning: A Comparative Study of Kuveyt Türk, VakıfBank, and İş Bankası

---

## Overview

This repository implements a full research pipeline that:

1. Collects customer complaints for three Turkish banks from Şikayetvar
2. Cleans and preprocesses Turkish complaint text
3. Discovers latent complaint topics with **LDA**
4. Predicts complaint resolution (`is_resolved`) with **TF-IDF + classical ML**
5. Tests five research hypotheses with comparative statistical analysis

| Bank | Type | Role in study |
|------|------|----------------|
| **VakıfBank** | Public bank | Public-sector customer experience |
| **İş Bankası** | Private commercial bank | Private-sector baseline |
| **Kuveyt Türk** | Participation (Islamic) bank | Participation banking comparison |

**Primary analysis year in code:** 2026 (see `config.py` and `data/raw/*_2026.csv`).  
Historical 2025 scrapes are retained under `sikayetvar-web-scraper/`.

---

## Research hypotheses

| ID | Hypothesis | Result (current run) |
|----|------------|----------------------|
| **H1** | Complaint topics differ significantly across banks | Supported |
| **H2** | Certain complaint categories raise non-resolution likelihood | Supported |
| **H3** | ML models can predict resolution with ≥ 70% accuracy | Partially supported |
| **H4** | Complaint volume shows significant temporal trends | Partially supported |
| **H5** | Resolution status and satisfaction are positively related | Supported |

Detailed evidence: [`results/reports/hipotez_sonuclari.csv`](results/reports/hipotez_sonuclari.csv)

---

## Pipeline

```text
┌─────────────────┐
│  Web scraping   │  sikayetvar-web-scraper/
└────────┬────────┘
         ▼
┌─────────────────┐
│ adim1_kurulum   │  folders, NLTK, stop words, CSV placement
└────────┬────────┘
         ▼
┌─────────────────┐
│ adim2_eda       │  exploratory analysis & quality checks
└────────┬────────┘
         ▼
┌─────────────────┐
│ adim3_onisleme  │  Turkish NLP preprocessing
└────────┬────────┘
         ▼
┌─────────────────┐
│ adim4_lda       │  topic modeling (gensim LDA + pyLDAvis)
└────────┬────────┘
         ▼
┌─────────────────┐
│ adim5_tfidf     │  TF-IDF + LR / RF / SVM
└────────┬────────┘
         ▼
┌─────────────────┐
│ adim6_karsilas… │  hypothesis tests & dashboards
└─────────────────┘
```

| Step | Script | Output highlights |
|------|--------|-------------------|
| 1 | `adim1_kurulum.py` | Project dirs, NLTK data, Turkish stop-word list |
| 2 | `adim2_eda.py` | EDA figures, summary stats |
| 3 | `adim3_onisleme.py` | `data/processed/veri_temiz.csv` |
| 4 | `adim4_lda.py` | LDA models, `veri_lda.csv`, pyLDAvis HTML |
| 5 | `adim5_tfidf.py` | Classifiers, `model_sonuclari.csv`, ROC/confusion plots |
| 6 | `adim6_karsilastirma.py` | `hipotez_sonuclari.csv`, comparison dashboards |

Central configuration lives in [`config.py`](config.py) (paths, LDA search range, TF-IDF size, train/test split, H3 threshold).

---

## Dataset

### Analysis inputs (`data/raw/`)

| File | Approx. rows | Description |
|------|--------------|-------------|
| `vakifbank_2026.csv` | ~800 | VakıfBank complaints |
| `is-bankasi_2026.csv` | ~3,800 | İş Bankası complaints |
| `kuveyt-turk_2026.csv` | ~1,800 | Kuveyt Türk complaints |

**Typical columns:** `operator`, `id`, `date`, `username`, `title`, `full_text`, `company_reply`, `is_resolved`, `satisfaction`, `keywords`, `view_count`, `upvote_count`, `url`, …

**Target variable:** `is_resolved` (resolution status).  
**Satisfaction policy:** median is used (not mean) because scores are heavily skewed — see comments in `config.py`.

### Processed artifacts

- `data/processed/veri_ham_birlesmis.csv` — merged raw data  
- `data/processed/veri_temiz.csv` — cleaned / stemmed text  
- `data/processed/veri_lda.csv` — LDA topic assignments  

### Scraping archives

Larger historical dumps (2025 & 2026 CSV/XLSX) live in [`sikayetvar-web-scraper/`](sikayetvar-web-scraper/).

---

## Tech stack

| Area | Tools |
|------|--------|
| Language | Python 3.12 |
| Data | pandas, numpy, scipy, openpyxl |
| NLP | NLTK, snowballstemmer (Turkish), custom banking stop words |
| Topic modeling | gensim LDA, CoherenceModel, pyLDAvis |
| ML | scikit-learn — TfidfVectorizer, LogisticRegression, RandomForest, LinearSVC |
| Visualization | matplotlib, seaborn, wordcloud |
| Persistence | joblib |
| Scraping | requests, BeautifulSoup, lxml |

> **Note:** `requirements.txt` also lists Hugging Face / PyTorch packages (BERTurk path). The current `adim1`–`adim6` pipeline uses classical NLP + sklearn; those deep-learning deps are optional for future embedding experiments.

---

## Getting started

### 1. Clone

```bash
git clone https://github.com/BetulAlbayrak1999/Analyzing-Banks-Reviews-Sikayetvar-ML-Text-Mining.git
cd Analyzing-Banks-Reviews-Sikayetvar-ML-Text-Mining
```

### 2. Create a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For scrapers only (lighter install):

```bash
pip install requests beautifulsoup4 lxml pandas openpyxl
```

### 4. Place raw data

Ensure these files exist (already present if you cloned with data):

```text
data/raw/vakifbank_2026.csv
data/raw/is-bankasi_2026.csv
data/raw/kuveyt-turk_2026.csv
```

Alternatively, put scraper outputs at the repo root as `sikayetvar_*.csv` and run `adim1_kurulum.py` to copy/normalize them into `data/raw/`.

### 5. Run the analysis pipeline

From the repository root, in order:

```bash
python adim1_kurulum.py
python adim2_eda.py
python adim3_onisleme.py
python adim4_lda.py
python adim5_tfidf.py
python adim6_karsilastirma.py
```

Trained models are written under `models/` (gitignored — regenerate with steps 4–5).  
Figures and reports appear under `results/`.

---

## Scraping (optional)

Scrapers collect Şikayetvar complaints and write CSV/XLSX. Runs can take **hours** depending on date range and site load.

```bash
cd sikayetvar-web-scraper
python get_raw_data_vakifbank.py
python get_raw_data_isbank.py
python get_raw_data_kuveytturk.py
```

Respect the website’s terms of use, robots rules, and rate limits. Prefer reusing published CSVs in this repo when possible.

---

## Project structure

```text
.
├── adim1_kurulum.py          # Setup
├── adim2_eda.py              # EDA
├── adim3_onisleme.py         # Turkish preprocessing
├── adim4_lda.py              # LDA topic modeling
├── adim5_tfidf.py            # TF-IDF + classifiers
├── adim6_karsilastirma.py    # Comparative hypothesis tests
├── config.py                 # Paths & hyperparameters
├── requirements.txt
├── data/
│   ├── raw/                  # Input CSVs (2026)
│   ├── processed/            # Cleaned & LDA-enriched data
│   └── turkce_stop_words*.txt
├── models/                   # Generated (gitignored)
├── results/
│   ├── figures/              # PNG charts
│   ├── lda/                  # Topics, pyLDAvis HTML
│   └── reports/              # Metrics & hypothesis CSVs
├── sikayetvar-web-scraper/   # Scrapers + archive dumps

```

---

## Selected results (illustrative)

Results below reflect the current committed reports; re-running the pipeline may change numbers slightly due to stochastic LDA/ML steps (seeded where applicable via `RANDOM_STATE = 42`).

### Resolution rates (EDA summary)

| Bank | Resolution rate (approx.) |
|------|---------------------------|
| VakıfBank | ~54% |
| Kuveyt Türk | ~47% |
| İş Bankası | ~26% |

### LDA — optimal topic counts

| Bank | Optimal *k* | Coherence (approx.) |
|------|-------------|---------------------|
| VakıfBank | 6 | ~0.40 |
| İş Bankası | 4 | ~0.41 |
| Kuveyt Türk | 8 | ~0.35 |

Interactive topic maps: `results/lda/ldavis_*.html`

### Best classification accuracies (sample)

| Bank | Strong model | Accuracy (approx.) |
|------|--------------|--------------------|
| İş Bankası | Linear SVM | ~0.78 |
| VakıfBank | Random Forest | ~0.74 |
| Kuveyt Türk | Random Forest | ~0.66 |

Full table: [`results/reports/model_sonuclari.csv`](results/reports/model_sonuclari.csv)

---

## Keywords

Metin madenciliği · NLP · Makine öğrenmesi · Bankacılık · Müşteri şikayetleri · LDA · TF-IDF · Sınıflandırma · Karşılaştırmalı analiz · Şikayetvar

---

## Citation

If you use this repository in academic work, please cite it as:

```bibtex
@software{albayrak_banks_sikayetvar_nlp,
  author  = {Albayrak, Betül},
  title   = {Analyzing Banks Reviews — Şikayetvar ML Text Mining},
  year    = {2026},
  url     = {https://github.com/BetulAlbayrak1999/Analyzing-Banks-Reviews-Sikayetvar-ML-Text-Mining},
  note    = {Marmara University thesis project}
}
```

A machine-readable citation is also available in [`CITATION.cff`](CITATION.cff).

---

## Ethics & data notice

- Data are collected from a **public** complaint platform.
- Complaints may contain personal information; do not redistribute scraped dumps beyond research use without review.
- This project does not claim affiliation with Şikayetvar or the banks studied.
- Follow applicable research ethics guidance at your institution.

---

## Contributing

Contributions that improve reproducibility, documentation, or analysis quality are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

This project is released under the [MIT License](LICENSE).

---

## Author

**Betül Albayrak**  
Marmara University — thesis research repository  
GitHub: [@BetulAlbayrak1999](https://github.com/BetulAlbayrak1999)
