# ScreenLot Demo Data

This folder contains the lightweight dataset bundle that ships with the public ScreenLot repository.

## Purpose

- give the Streamlit app a working catalog immediately after clone
- avoid requiring the full MovieLens 32M download for first-run testing
- keep recommendation, EDA, and benchmark pages from loading into an empty state

## Contents

- `screenlot-demo/ratings.csv`: sampled MovieLens ratings for a compact demo profile
- `screenlot-demo/movies.csv`: filtered ScreenLot catalog rows with metadata used by the app
- `screenlot-demo/links.csv`: filtered MovieLens links for the same movie subset

## Switching to full data

When you want the full staged dataset instead of the packaged demo bundle, set:

`SCREENLOT_DATA_DIR=/absolute/path/to/data/raw/movielens/ml-32m`
