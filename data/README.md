# ScreenLot Data Setup

ScreenLot now has an open-data pipeline based on:

- MovieLens 32M as the ratings and interaction source
- Wikidata as the explainable metadata enrichment source

## Local layout

- `data/app/screenlot-demo/`
  Stores the lightweight tracked demo bundle that powers the public repo out of the box.
- `data/raw/movielens/`
  Stores the official MovieLens 32M archive, checksum, README, and extracted CSV files.
- `data/interim/wikidata/`
  Stores incremental Wikidata enrichment outputs.
- `data/processed/screenlot/`
  Stores the merged ScreenLot-ready catalog and summary tables.

`data/raw/`, `data/interim/`, and `data/processed/` are intentionally ignored by git because the files are large and generated locally. The `data/app/screenlot-demo/` folder is the exception because it is a compact deployment-ready sample bundle.

## Pipeline commands

From the repository root:

1. Stage MovieLens 32M

   `python3 scripts/stage_movielens_32m.py`

2. Enrich MovieLens titles from Wikidata

   `python3 scripts/enrich_wikidata_movies.py`

   Useful for an initial smoke test:

   `python3 scripts/enrich_wikidata_movies.py --limit 500`

3. Build the ScreenLot-ready catalog

   `python3 scripts/build_screenlot_catalog.py`

## Key generated outputs

- `data/processed/screenlot/catalog.csv`
- `data/processed/screenlot/top_movies.csv`
- `data/processed/screenlot/rating_distribution.csv`
- `data/processed/screenlot/user_activity_summary.csv`
- `data/processed/screenlot/dataset_profile.json`

## Notes

- The original notebook-era `movie_recommendation_data/` flow is still documented for backward compatibility.
- MovieLens 32M is much newer and larger than the original competition snapshot used in the notebooks.
- Wikidata enrichment is designed to be incremental so it can resume without starting over.
- Set `SCREENLOT_DATA_DIR` if you want the app to use a different dataset folder than the packaged demo bundle.
