# ScreenLot

ScreenLot is the product layer for the original movie recommendation project in this repository. The notebooks, presentation materials, and archived model artifacts are still here, but the repo now also includes a Streamlit app foundation that turns the project into a reusable product.

## What is in the repo now

- `app.py`: Streamlit entrypoint for ScreenLot.
- `screenlot/`: reusable app, data, EDA, and recommendation modules.
- `assets/`: extracted branding, contributor photos, and presentation-derived visuals.
- `notebooks/`: companion notebooks that explain the EDA and modeling work.
- `data/README.md`: instructions for staging the original project CSV files.
- `Model pickles.zip`: archived collaborative filtering artifacts from the original work.
- Original notebooks and presentation decks for reference.

## ScreenLot product sections

- Home
- Recommendation Engine
- EDA
- About
- Report and Conclusion
- Suggestions

## Recommendation approach

The current ScreenLot foundation uses a hybrid strategy:

- validation-tuned hybrid ranking from collaborative, metadata, and popularity signals
- a live model-selection layer that keeps the best validation-winning serving engine
- structured explanations for every recommendation
- persisted user feedback capture for likes, saves, and dismissals

This keeps the app useful in product form while still respecting the original modeling work from the notebooks.

## UI and product experience

The current ScreenLot UI now includes:

- a purple-ash and black brand system
- a custom ScreenLot banner asset in `assets/branding/screenlot-banner.svg`
- contributor imagery on the About page
- live model snapshot cards and benchmark visualizations
- quick-start profile buttons for recommendation prompts
- explanation chips and deeper reasoning details on recommendations
- persistent recommendation feedback stored locally outside the repo by default

## Open Data Pipeline

The repository now includes a reproducible open-data staging pipeline for the next version of ScreenLot:

- `scripts/stage_movielens_32m.py`
- `scripts/enrich_wikidata_movies.py`
- `scripts/build_screenlot_catalog.py`

The pipeline stages:

- raw MovieLens 32M ratings, movies, links, and tags
- incremental Wikidata movie enrichment for explainability
- processed ScreenLot-ready catalog and summary tables

Quick start:

1. `python3 scripts/stage_movielens_32m.py`
2. `python3 scripts/enrich_wikidata_movies.py --limit 500`
3. `python3 scripts/build_screenlot_catalog.py`

## Modeling Pipeline

The repository now includes an offline benchmarking and artifact-generation script for ScreenLot:

- `python3 scripts/train_screenlot_models.py`

What it does:

- builds timestamp-based train, validation, and test splits
- benchmarks popularity, item-item collaborative filtering, matrix factorization, and the ScreenLot hybrid model
- reports ranking metrics such as `Recall@K`, `NDCG@K`, `MAP@K`, coverage, novelty, and diversity
- saves a trained ScreenLot artifact for the app to load automatically when present
- lets the saved artifact route recommendations through the best validation-winning ranker when it outperforms the explainable hybrid on top-line relevance

Default training is intentionally sample-sized for iteration speed. The `--max-rows` option now targets an approximate volume by randomly selecting complete user histories instead of taking the first rows in the CSV, which keeps the temporal evaluation much more trustworthy. Use `--max-rows 0` and a larger `--max-users` value when you want a more expensive full-dataset run.

## Run the app

1. Install the requirements:

   `pip install -r requirements.txt`

2. Add the original project CSV files to `movie_recommendation_data/` in the repo root.

3. Launch ScreenLot:

   `streamlit run app.py`

## Companion notebooks

Two notebooks have been added to explain the current productized version of the project:

- `notebooks/ScreenLot_Modeling_Workbook.ipynb`
- `notebooks/ScreenLot_EDA_Workbook.ipynb`

These notebooks explain the modern modeling pipeline, benchmark results, open-data staging flow, and the EDA views that feed the Streamlit app.

## Runtime and deployment

ScreenLot now supports environment-driven paths so the same app can run locally or in a hosted deployment with mounted data and writable state:

- `SCREENLOT_DATA_DIR`: location of the ratings and movies CSV files
- `SCREENLOT_STATE_DIR`: writable directory for app state such as user feedback
- `SCREENLOT_FEEDBACK_LOG`: optional explicit path for the feedback log file
- `SCREENLOT_MODEL_ARTIFACT`: optional explicit path to the trained recommendation artifact
- `SCREENLOT_MODEL_CARD`: optional explicit path to the saved model-card JSON

Example:

```bash
export SCREENLOT_DATA_DIR=/mount/data/ml-32m
export SCREENLOT_STATE_DIR=/mount/state/screenlot
streamlit run app.py
```

User feedback from the recommendation page is persisted as JSON Lines so likes, saves, and dismissals survive app restarts.

## Data expectations

Minimum files for the first working experience:

- `movie_recommendation_data/train.csv`
- `movie_recommendation_data/movies.csv`

Recommended for a stronger recommendation engine and richer EDA:

- `movie_recommendation_data/imdb_data.csv`
- `movie_recommendation_data/links.csv`
- `movie_recommendation_data/test.csv`
- `movie_recommendation_data/genome_scores.csv`
- `movie_recommendation_data/genome_tags.csv`

## Notes

- The app now uses the ScreenLot branding and contributor assets extracted from the project presentation deck.
- The current UI theme is based on purple ash and black to give the product a more cinematic streaming feel.
- The original repo snapshot included notebooks only; this pass begins the conversion into a maintainable application.
- The archived collaborative models remain optional at runtime because some environments may not have the dependencies needed to unpickle them.
- The open-data pipeline is designed so we can move off the older Kaggle-era snapshot and build the next model version on MovieLens 32M plus Wikidata.
