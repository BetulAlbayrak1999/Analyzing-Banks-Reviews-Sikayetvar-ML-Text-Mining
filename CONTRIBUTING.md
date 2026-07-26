# Contributing

Thank you for your interest in improving this thesis research repository.

## Ways to contribute

- Fix bugs in the analysis or scraper scripts
- Improve documentation (README, comments, docstrings)
- Add tests or reproducibility checks
- Suggest clearer visualizations or reporting outputs
- Share issues with environment setup on different platforms

## Development setup

1. Fork and clone the repository
2. Create a virtual environment with **Python 3.12**
3. Install dependencies: `pip install -r requirements.txt`
4. Ensure sample CSVs exist under `data/raw/` (or scrape new data carefully)

## Workflow

1. Create a feature branch from `main` or `Dev-2025` (match the branch you intend to PR into)
2. Make focused changes — prefer small PRs over large mixed commits
3. Run the relevant pipeline step(s) to verify your change
4. Open a pull request using the PR template

## Coding guidelines

- Keep the sequential `adim1` → `adim6` design unless there is a strong reason to refactor
- Prefer updating `config.py` for paths and hyperparameters instead of hard-coding values
- Do not commit secrets, API keys, or local virtual environments
- Do not commit Word lock files (`~$*`) or editor temporary files
- Avoid committing large regenerated artifacts unless necessary; `models/` is gitignored by design
- Match existing naming style (Turkish step names / English technical terms as already used)

## Scraping ethics

If you change scrapers:

- Keep polite delays between requests
- Do not add aggressive parallel scraping by default
- Document any change to date windows or output schemas
- Prefer incremental / resume-friendly collection when possible

## Reporting issues

Use GitHub Issues with:

- What you expected vs. what happened
- OS, Python version, and key package versions
- Minimal steps to reproduce
- Relevant logs or traceback (redact personal data from complaint text)

## Academic integrity

This repository supports a university thesis. Please:

- Do not misrepresent results or remove statistical caveats
- Attribute prior work when adapting methods
- Discuss substantial methodological changes in the PR description

## License

By contributing, you agree that your contributions will be licensed under the MIT License covering this project.
