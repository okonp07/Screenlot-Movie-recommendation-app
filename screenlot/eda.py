from __future__ import annotations

from typing import Any

try:
    import pandas as pd
    import plotly.express as px
except ImportError:  # pragma: no cover - handled at runtime in the app.
    pd = None
    px = None

from .model_summary import humanize_model_name, leaderboard_rows
from .styles import plotly_layout


PALETTE = ["#d9a441", "#4bb3c3", "#d45769", "#7cd38b", "#f4ead5"]


def _require_runtime() -> None:
    if pd is None or px is None:
        raise RuntimeError("plotly and pandas are required for the ScreenLot EDA views.")


def _apply_visual_theme(figure: Any, theme_mode: str, **extra_layout: Any) -> Any:
    figure.update_layout(**plotly_layout(theme_mode))
    if extra_layout:
        figure.update_layout(**extra_layout)
    return figure


def ratings_distribution(train: Any, theme_mode: str = "dark") -> Any:
    _require_runtime()
    figure = px.histogram(
        train,
        x="rating",
        nbins=10,
        title="Distribution of User Ratings",
        color_discrete_sequence=[PALETTE[0]],
    )
    return _apply_visual_theme(figure, theme_mode, bargap=0.08)


def movies_per_year(catalog: Any, theme_mode: str = "dark") -> Any:
    _require_runtime()
    frame = (
        catalog.dropna(subset=["release_year"])
        .groupby("release_year", as_index=False)
        .agg(movie_count=("movie_id", "nunique"))
        .sort_values("release_year")
    )
    figure = px.line(
        frame,
        x="release_year",
        y="movie_count",
        markers=True,
        title="Movies Released Per Year",
    )
    figure.update_traces(line_color=PALETTE[1])
    return _apply_visual_theme(figure, theme_mode)


def genre_distribution(catalog: Any, top_n: int = 12, theme_mode: str = "dark") -> Any:
    _require_runtime()
    frame = (
        catalog.assign(genres=catalog["genres"].fillna("").str.split("|"))
        .explode("genres")
        .query("genres != ''")
        .groupby("genres", as_index=False)
        .agg(movie_count=("movie_id", "nunique"))
        .sort_values("movie_count", ascending=False)
        .head(top_n)
    )
    figure = px.bar(
        frame,
        x="movie_count",
        y="genres",
        orientation="h",
        title="Most Common Genres in the Catalog",
        color="movie_count",
        color_continuous_scale=["#2e425c", "#d9a441"],
    )
    return _apply_visual_theme(figure, theme_mode, coloraxis_showscale=False)


def top_rated_movies(
    catalog: Any,
    min_ratings: int = 50,
    top_n: int = 10,
    theme_mode: str = "dark",
) -> Any:
    _require_runtime()
    frame = (
        catalog.loc[catalog["rating_count"].fillna(0) >= min_ratings, ["title", "mean_rating", "rating_count"]]
        .dropna(subset=["mean_rating"])
        .sort_values(["mean_rating", "rating_count"], ascending=[False, False])
        .head(top_n)
    )
    figure = px.bar(
        frame,
        x="mean_rating",
        y="title",
        orientation="h",
        title="Highest Rated Movies",
        color="rating_count",
        color_continuous_scale=["#4bb3c3", "#d9a441"],
    )
    return _apply_visual_theme(figure, theme_mode, coloraxis_showscale=False)


def top_directors(catalog: Any, top_n: int = 10, theme_mode: str = "dark") -> Any | None:
    _require_runtime()

    director_column = next(
        (column for column in catalog.columns if column in {"director", "directors"}),
        None,
    )
    if director_column is None:
        return None

    frame = (
        catalog.dropna(subset=[director_column])
        .assign(**{director_column: catalog[director_column].astype(str).str.split("|")})
        .explode(director_column)
    )
    frame[director_column] = frame[director_column].str.strip()
    frame = (
        frame.query(f"{director_column} != ''")
        .groupby(director_column, as_index=False)
        .agg(movie_count=("movie_id", "nunique"))
        .sort_values("movie_count", ascending=False)
        .head(top_n)
    )

    figure = px.bar(
        frame,
        x="movie_count",
        y=director_column,
        orientation="h",
        title="Most Frequent Directors",
        color_discrete_sequence=[PALETTE[2]],
    )
    return _apply_visual_theme(figure, theme_mode)


def benchmark_metric_bars(
    model_card: dict[str, Any] | None,
    split: str = "test",
    theme_mode: str = "dark",
) -> Any | None:
    _require_runtime()
    rows = leaderboard_rows(model_card, split=split)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    frame["Model"] = frame["model"].apply(humanize_model_name)
    melted = frame.melt(
        id_vars=["Model"],
        value_vars=["recall@20", "ndcg@20", "map@20"],
        var_name="Metric",
        value_name="Score",
    )
    figure = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        title=f"{split.title()} Ranking Metrics",
        color_discrete_sequence=PALETTE[:3],
    )
    return _apply_visual_theme(figure, theme_mode, legend_title_text="")


def benchmark_tradeoff_scatter(
    model_card: dict[str, Any] | None,
    split: str = "test",
    theme_mode: str = "dark",
) -> Any | None:
    _require_runtime()
    rows = leaderboard_rows(model_card, split=split)
    if not rows:
        return None
    frame = pd.DataFrame(rows)
    if frame.empty:
        return None
    frame["Model"] = frame["model"].apply(humanize_model_name)
    figure = px.scatter(
        frame,
        x="coverage",
        y="recall@20",
        size="novelty",
        color="Model",
        hover_data=["ndcg@20", "map@20", "evaluated_users"],
        title=f"{split.title()} Coverage vs Recall",
        color_discrete_sequence=PALETTE,
    )
    return _apply_visual_theme(figure, theme_mode)


def feedback_actions_chart(feedback_frame: Any, theme_mode: str = "dark") -> Any | None:
    _require_runtime()
    if feedback_frame is None or getattr(feedback_frame, "empty", True):
        return None
    frame = (
        feedback_frame.groupby("action", as_index=False)
        .size()
        .sort_values("size", ascending=False)
    )
    if frame.empty:
        return None
    frame["action"] = frame["action"].replace(
        {"like": "Liked", "save": "Saved", "not_for_me": "Not for me"}
    )
    figure = px.bar(
        frame,
        x="action",
        y="size",
        title="User Feedback Actions",
        color="action",
        color_discrete_sequence=PALETTE[:3],
    )
    return _apply_visual_theme(figure, theme_mode, showlegend=False)


def feedback_top_titles_chart(
    feedback_frame: Any,
    top_n: int = 8,
    theme_mode: str = "dark",
) -> Any | None:
    _require_runtime()
    if feedback_frame is None or getattr(feedback_frame, "empty", True):
        return None
    frame = (
        feedback_frame.groupby(["title", "action"], as_index=False)
        .size()
        .sort_values("size", ascending=False)
        .head(top_n)
    )
    if frame.empty:
        return None
    frame["action"] = frame["action"].replace(
        {"like": "Liked", "save": "Saved", "not_for_me": "Not for me"}
    )
    figure = px.bar(
        frame,
        x="size",
        y="title",
        color="action",
        orientation="h",
        title="Top Feedback Titles",
        color_discrete_sequence=PALETTE[:3],
    )
    return _apply_visual_theme(figure, theme_mode)
