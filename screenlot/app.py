from __future__ import annotations

from pathlib import Path
from typing import Any
import zipfile

import streamlit as st

from . import eda
from .content import (
    ABOUT_PROJECT,
    APP_NAME,
    APP_TAGLINE,
    BENCHMARK_RMSE,
    CONTRIBUTORS,
    DATAPORT_WORDMARK,
    DEFAULT_DATA_DIR,
    ECONOMIC_VALUE,
    FEATURE_PILLARS,
    MODEL_ARCHIVE,
    MODEL_STRATEGY,
    NAVIGATION,
    ORGANIZATION,
    REPORT_SUMMARY,
    SCREENLOT_BANNER,
    SCREENLOT_LOGO,
    STREAMLIT_CONCEPT,
    SUGGESTION_ITEMS,
    SUPERVISOR,
)
from .data import (
    available_genres,
    build_catalog,
    dataset_metrics,
    dataset_status,
    load_data_bundle,
    missing_required_files,
)
from .feedback import append_feedback, feedback_summary, load_feedback_frame
from .model_summary import (
    humanize_model_name,
    leaderboard_rows,
    load_model_card,
    model_snapshot,
)
from .recommender import (
    ScreenLotHybridRecommender,
    load_archived_collaborative_model,
    load_trained_screenlot_artifact,
)
from .runtime import FEEDBACK_LOG_PATH, STATE_DIR
from .styles import GLOBAL_CSS


def _safe_image(path: Path) -> str | None:
    return str(path) if path.exists() else None


def _render_banner_image() -> None:
    banner_path = _safe_image(SCREENLOT_BANNER)
    if banner_path:
        st.image(banner_path, use_container_width=True)


def _apply_page_config() -> None:
    page_icon = _safe_image(SCREENLOT_LOGO)
    st.set_page_config(
        page_title=APP_NAME,
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(f"<style>{GLOBAL_CSS}</style>", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def _load_bundle(data_dir: str):
    bundle = load_data_bundle(Path(data_dir))
    catalog = build_catalog(bundle)
    return bundle, catalog


@st.cache_data(show_spinner=False)
def _load_saved_model_card():
    return load_model_card()


@st.cache_resource(show_spinner=False)
def _load_recommender(data_dir: str):
    bundle, catalog = _load_bundle(data_dir)
    trained_artifact, trained_status = load_trained_screenlot_artifact()
    if trained_artifact is not None:
        return trained_artifact, trained_status
    collaborative_model, collaborative_status = load_archived_collaborative_model(MODEL_ARCHIVE)
    engine = ScreenLotHybridRecommender(catalog, collaborative_model=collaborative_model)
    return engine, collaborative_status


def _metric_card(label: str, value: str) -> str:
    return (
        "<div class='metric-card'>"
        f"<div class='metric-label'>{label}</div>"
        f"<div class='metric-value'>{value}</div>"
        "</div>"
    )


def _format_metric_value(value: float | int | None, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.{digits}f}"


def _score_band(score: float) -> str:
    if score >= 0.78:
        return "Excellent match"
    if score >= 0.62:
        return "Strong match"
    if score >= 0.48:
        return "Good match"
    return "Discovery pick"


def _leaderboard_table_rows(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    table_rows: list[dict[str, str]] = []
    for row in rows:
        table_rows.append(
            {
                "Model": humanize_model_name(str(row.get("model", ""))),
                "Recall@10": _format_metric_value(row.get("recall@10")),
                "Recall@20": _format_metric_value(row.get("recall@20")),
                "NDCG@10": _format_metric_value(row.get("ndcg@10")),
                "NDCG@20": _format_metric_value(row.get("ndcg@20")),
                "MAP@20": _format_metric_value(row.get("map@20")),
                "Coverage": _format_metric_value(row.get("coverage")),
                "Novelty": _format_metric_value(row.get("novelty")),
                "Users": f"{int(row.get('evaluated_users', 0)):,}",
            }
        )
    return table_rows


def _feedback_labels(action: str) -> str:
    return {
        "like": "Liked",
        "save": "Saved",
        "not_for_me": "Not for me",
    }.get(action, action.replace("_", " ").title())


def _badge_markup(tags: str | None) -> str:
    if not tags:
        return ""
    parts = [part.strip() for part in str(tags).split("|") if part.strip()]
    if not parts:
        return ""
    return "<div class='explanation-strip'>" + "".join(
        f"<div class='explanation-chip'>{part}</div>" for part in parts
    ) + "</div>"


def _render_feedback_snapshot(summary: dict[str, Any]) -> None:
    total_events = int(summary.get("total_events", 0))
    action_counts = summary.get("action_counts", {})
    metric_left, metric_mid, metric_right, metric_last = st.columns(4)
    metric_left.metric("Feedback Events", f"{total_events:,}")
    metric_mid.metric("Likes", f"{int(action_counts.get('like', 0)):,}")
    metric_right.metric("Saves", f"{int(action_counts.get('save', 0)):,}")
    metric_last.metric("Not for me", f"{int(action_counts.get('not_for_me', 0)):,}")


def _snapshot_comparison_text(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return "Train the ScreenLot artifact to expose live model-selection context."
    serving_row = snapshot["serving_row"]
    runner_up = snapshot.get("runner_up_row")
    validation_row = snapshot.get("validation_row") or {}
    if not runner_up:
        return (
            f"ScreenLot is serving {snapshot['model_label']} based on validation results "
            f"across {int(validation_row.get('evaluated_users', 0)):,} users."
        )
    recall_delta = float(serving_row.get("recall@20", 0.0)) - float(runner_up.get("recall@20", 0.0))
    ndcg_delta = float(serving_row.get("ndcg@20", 0.0)) - float(runner_up.get("ndcg@20", 0.0))
    runner_up_label = humanize_model_name(str(runner_up.get("model", "")))
    if recall_delta >= 0 and ndcg_delta >= 0:
        return (
            f"ScreenLot is serving {snapshot['model_label']} because it beat "
            f"{runner_up_label} on validation and still leads the latest test benchmark."
        )
    return (
        f"ScreenLot is serving {snapshot['model_label']} because it won validation, "
        f"even though {runner_up_label} remains close on the latest test benchmark."
    )


def _starter_profiles(catalog) -> list[dict[str, Any]]:
    if catalog is None or getattr(catalog, "empty", True):
        return []

    profile_specs = [
        ("Family Adventure", ["Adventure", "Animation", "Children"]),
        ("Comedy Night", ["Comedy"]),
        ("Thriller Run", ["Thriller", "Crime", "Mystery"]),
        ("Romance Drama", ["Romance", "Drama"]),
    ]

    working = catalog.copy()
    working["genres_text"] = working["genres"].fillna("").astype(str)
    working["mean_rating"] = working["mean_rating"].fillna(0.0)
    working["rating_count"] = working["rating_count"].fillna(0)

    profiles: list[dict[str, Any]] = []
    for label, genres in profile_specs:
        mask = working["genres_text"].apply(
            lambda value: any(genre in str(value).split("|") for genre in genres)
        )
        subset = working.loc[mask].sort_values(
            ["rating_count", "mean_rating"],
            ascending=[False, False],
        )
        titles = subset["title"].dropna().astype(str).drop_duplicates().head(3).tolist()
        if len(titles) < 2:
            continue
        profiles.append(
            {
                "label": label,
                "genres": genres,
                "titles": titles,
                "summary": ", ".join(titles[:3]),
            }
        )
    return profiles


def _render_model_snapshot(snapshot: dict[str, Any] | None) -> None:
    st.markdown("### Live Model Snapshot")
    if not snapshot:
        st.info("Train the ScreenLot artifact to show live benchmark details here.")
        return

    serving_row = snapshot["serving_row"]
    validation_row = snapshot.get("validation_row") or {}
    training_config = snapshot.get("training_config", {})
    metric_left, metric_mid, metric_right, metric_last = st.columns(4)
    metric_left.metric("Serving Model", snapshot["selected_model_label"])
    metric_mid.metric("Recall@20", _format_metric_value(serving_row.get("recall@20")))
    metric_right.metric("NDCG@20", _format_metric_value(serving_row.get("ndcg@20")))
    metric_last.metric("Evaluated Users", f"{int(serving_row.get('evaluated_users', 0)):,}")

    elapsed_seconds = float(training_config.get("elapsed_seconds", 0.0) or 0.0)
    st.markdown(
        (
            "<div class='section-shell'>"
            "<h3 class='section-title'>Why this engine is live</h3>"
            f"<div class='section-copy'>{_snapshot_comparison_text(snapshot)}</div>"
            f"<div class='small-note' style='margin-top: 1rem;'>"
            f"Score strategy: {snapshot.get('score_strategy', 'unknown')}. "
            f"Validation users: {int(validation_row.get('evaluated_users', 0)):,}. "
            f"Training target: {int(training_config.get('max_rows', 0)):,} ratings. "
            f"Last training run: {elapsed_seconds / 60:.1f} minutes."
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _store_recommendation_results(
    results,
    favorite_titles: list[str],
    selected_genres: list[str],
    min_year: int | None,
) -> None:
    st.session_state["recommendation_results"] = results.to_dict(orient="records")
    st.session_state["recommendation_context"] = {
        "favorite_titles": list(favorite_titles),
        "selected_genres": list(selected_genres),
        "min_year": min_year,
    }


def _clear_recommendation_state(years) -> None:
    st.session_state["favorite_titles"] = []
    st.session_state["selected_genres"] = []
    st.session_state["top_n"] = 10
    st.session_state["user_id_input"] = ""
    st.session_state["recommendation_results"] = []
    st.session_state["recommendation_context"] = {}
    if "min_year" in st.session_state and not years.empty:
        st.session_state["min_year"] = int(years.min())


def _submit_feedback(
    record: dict[str, Any],
    rank: int,
    action: str,
    snapshot: dict[str, Any] | None,
) -> None:
    context = st.session_state.get("recommendation_context", {})
    append_feedback(
        action=action,
        movie_id=int(record.get("movie_id", 0)),
        title=str(record.get("title", "Untitled")),
        score=float(record.get("score", 0.0)),
        reason=str(record.get("reason", "")),
        favorite_titles=list(context.get("favorite_titles", [])),
        selected_model=(snapshot or {}).get("model_label", "ScreenLot"),
        rank=rank,
    )
    st.session_state["feedback_notice"] = f"{_feedback_labels(action)}: {record.get('title', 'title saved')}"
    st.rerun()


def _render_benchmark_table(model_card: dict[str, Any] | None) -> None:
    if not model_card:
        st.info("Benchmark tables will appear here once the ScreenLot artifact has been trained.")
        return

    validation_tab, test_tab = st.tabs(["Validation Leaderboard", "Test Leaderboard"])
    with validation_tab:
        st.dataframe(
            _leaderboard_table_rows(leaderboard_rows(model_card, split="validation")),
            use_container_width=True,
            hide_index=True,
        )
    with test_tab:
        st.dataframe(
            _leaderboard_table_rows(leaderboard_rows(model_card, split="test")),
            use_container_width=True,
            hide_index=True,
        )


def _render_model_performance_panel(model_card: dict[str, Any] | None) -> None:
    if not model_card:
        st.info("Train the ScreenLot artifact to unlock the model-performance dashboard.")
        return

    metric_chart = eda.benchmark_metric_bars(model_card, split="test")
    tradeoff_chart = eda.benchmark_tradeoff_scatter(model_card, split="test")
    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        if metric_chart is not None:
            st.plotly_chart(metric_chart, use_container_width=True)
    with chart_right:
        if tradeoff_chart is not None:
            st.plotly_chart(tradeoff_chart, use_container_width=True)

    _render_benchmark_table(model_card)


def _render_sidebar(
    data_dir: Path,
    readiness: dict[str, bool],
    snapshot: dict[str, Any] | None,
    feedback_snapshot: dict[str, Any],
) -> str:
    with st.sidebar:
        logo_path = _safe_image(SCREENLOT_LOGO)
        if logo_path:
            st.image(logo_path, width=150)

        st.markdown("### ScreenLot")
        st.caption("Cinematic movie discovery from the original recommendation project.")
        page = st.radio("Navigate", NAVIGATION, index=0)

        st.markdown("### Data Folder")
        st.code(str(data_dir), language="text")

        st.markdown("### Data Readiness")
        for file_name, is_ready in readiness.items():
            status = "Ready" if is_ready else "Missing"
            st.caption(f"{status}: {file_name}")

        st.markdown("### Assets")
        st.caption("Branding, contributor images, and presentation materials are now wired into the repo.")

        st.markdown("### Model Status")
        if snapshot:
            serving_row = snapshot["serving_row"]
            st.caption(snapshot["model_label"])
            st.caption(
                f"Recall@20: {_format_metric_value(serving_row.get('recall@20'))} | "
                f"NDCG@20: {_format_metric_value(serving_row.get('ndcg@20'))}"
            )
        else:
            st.caption("Train a ScreenLot artifact to show the live engine summary here.")

        st.markdown("### Product Memory")
        st.caption(
            f"Feedback events: {int(feedback_snapshot.get('total_events', 0)):,}"
        )
        st.caption(f"State path: {STATE_DIR}")

    return page


def _render_home(
    metrics: dict[str, int] | None,
    collaborative_status: str,
    snapshot: dict[str, Any] | None,
    feedback_snapshot: dict[str, Any],
) -> None:
    _render_banner_image()

    hero_left, hero_right = st.columns((1.15, 0.85), gap="large")

    with hero_left:
        logo_path = _safe_image(SCREENLOT_LOGO)
        if logo_path:
            st.image(logo_path, width=88)
        st.markdown(
            (
                "<div class='hero-shell'>"
                "<div class='kicker'>ScreenLot</div>"
                f"<div class='hero-title'>{APP_NAME} turns this recommendation project into a premium discovery experience.</div>"
                f"<div class='hero-copy'>{APP_TAGLINE}</div>"
                "<div class='pill-strip'>"
                + "".join(f"<div class='pill'>{pillar}</div>" for pillar in FEATURE_PILLARS)
                + "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with hero_right:
        st.markdown(
            (
                "<div class='section-shell'>"
                "<h3 class='section-title'>Product Direction</h3>"
                "<div class='section-copy'>"
                "ScreenLot now blends modern recommendation UX, benchmark visibility, and explainability so the "
                "project reads like a product instead of a notebook handoff."
                "</div>"
                "<div class='insight-grid' style='margin-top: 1rem;'>"
                "<div class='status-callout'><strong>Brand</strong><br/>Purple ash, black glass, cinematic depth.</div>"
                "<div class='status-callout'><strong>Experience</strong><br/>Streaming-style discovery with feedback memory.</div>"
                "</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
        preview_path = _safe_image(STREAMLIT_CONCEPT)
        if preview_path:
            st.image(preview_path, use_container_width=True)

    st.markdown("### Product Snapshot")
    if metrics:
        st.markdown(
            (
                "<div class='metric-strip'>"
                f"{_metric_card('Catalog Size', f'{metrics['movies']:,}')}"
                f"{_metric_card('Ratings', f'{metrics['ratings']:,}')}"
                f"{_metric_card('Users', f'{metrics['users']:,}')}"
                "</div>"
            ),
            unsafe_allow_html=True,
        )
    else:
        st.info(
            "Run the open-data pipeline so `ratings.csv` and `movies.csv` are staged in "
            "`data/raw/movielens/ml-32m/`, then reload the app."
        )

    _render_model_snapshot(snapshot)
    if int(feedback_snapshot.get("total_events", 0)) > 0:
        st.markdown("### User Feedback Snapshot")
        _render_feedback_snapshot(feedback_snapshot)

    st.markdown(
        (
            "<div class='section-shell'>"
            "<h3 class='section-title'>Why this matters</h3>"
            f"<div class='section-copy'>{ECONOMIC_VALUE}</div>"
            f"<div class='small-note' style='margin-top: 1rem;'>Collaborative artifact status: {collaborative_status}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_recommendations(
    catalog,
    engine: Any | None,
    collaborative_status: str,
    snapshot: dict[str, Any] | None,
    model_card: dict[str, Any] | None,
    feedback_snapshot: dict[str, Any],
) -> None:
    st.markdown("## Recommendation Engine")
    st.write(
        "Pick a few titles, optionally narrow the genre or release window, and "
        "let the hybrid engine rank candidates."
    )

    if engine is None or catalog is None:
        st.warning(
            "The recommendation engine is waiting for local source data. Stage the "
            "MovieLens 32M pipeline and reload the app."
        )
        return

    if snapshot:
        serving_row = snapshot["serving_row"]
        top_left, top_mid, top_right = st.columns(3)
        top_left.metric("Live Engine", snapshot["selected_model_label"])
        top_mid.metric("Recall@20", _format_metric_value(serving_row.get("recall@20")))
        top_right.metric("NDCG@20", _format_metric_value(serving_row.get("ndcg@20")))
        st.caption(_snapshot_comparison_text(snapshot))

    starter_profiles = _starter_profiles(catalog)
    if starter_profiles:
        st.markdown("### Quick Start Profiles")
        profile_columns = st.columns(len(starter_profiles), gap="large")
        for index, profile in enumerate(starter_profiles):
            with profile_columns[index]:
                st.markdown(f"**{profile['label']}**")
                st.caption(profile["summary"])
                if st.button(
                    f"Use {profile['label']}",
                    key=f"starter_profile_{index}",
                    use_container_width=True,
                ):
                    st.session_state["favorite_titles"] = profile["titles"]
                    st.session_state["selected_genres"] = profile["genres"]
                    st.rerun()

    titles = sorted(catalog["title"].dropna().astype(str).unique().tolist())
    genres = available_genres(catalog)
    st.session_state.setdefault("favorite_titles", [])
    st.session_state.setdefault("selected_genres", [])
    st.session_state.setdefault("top_n", 10)
    st.session_state.setdefault("user_id_input", "")
    st.session_state.setdefault("recommendation_results", [])
    st.session_state.setdefault("recommendation_context", {})
    st.session_state.setdefault("feedback_notice", "")

    filter_left, filter_mid, filter_right = st.columns((1.3, 1, 1), gap="large")
    with filter_left:
        favorite_titles = st.multiselect(
            "Favorite movies",
            options=titles,
            placeholder="Search and choose at least two movies",
            key="favorite_titles",
        )
    with filter_mid:
        selected_genres = st.multiselect(
            "Genre filter",
            options=genres,
            placeholder="Optional genre focus",
            key="selected_genres",
        )
        top_n = st.slider("Recommendation count", min_value=5, max_value=20, key="top_n")
    with filter_right:
        years = catalog["release_year"].dropna().astype(int)
        min_year = None
        if not years.empty:
            st.session_state.setdefault("min_year", int(years.min()))
            min_year = st.slider(
                "Released after",
                min_value=int(years.min()),
                max_value=int(years.max()),
                key="min_year",
            )
        user_id_input = st.text_input(
            "Known user ID",
            help="Optional. Used only when the archived collaborative model can be loaded.",
            key="user_id_input",
        )

    st.caption(collaborative_status)
    _, action_right = st.columns((3, 1))
    with action_right:
        if st.button("Reset", use_container_width=True):
            _clear_recommendation_state(years)
            st.rerun()

    with st.expander("Live benchmark summary", expanded=False):
        if snapshot:
            st.write(
                f"ScreenLot is currently serving **{snapshot['model_label']}** based on the "
                "latest validation leaderboard."
            )
        _render_model_performance_panel(model_card)

    if st.button("Generate ScreenLot picks", use_container_width=True):
        if len(favorite_titles) < 2:
            st.warning("Choose at least two favorite titles so the engine can build a meaningful profile.")
            return

        user_id = int(user_id_input) if user_id_input.strip().isdigit() else None
        try:
            results = engine.recommend_from_titles(
                favorite_titles=favorite_titles,
                top_n=top_n,
                user_id=user_id,
                genre_filter=selected_genres,
                min_year=min_year,
            )
        except Exception as exc:  # pragma: no cover - surfaced directly in the app.
            st.error(f"Recommendation failed: {exc}")
            return

        if results.empty:
            st.warning("ScreenLot could not find matches for that combination. Try relaxing the genre or year filter.")
            return

        _store_recommendation_results(results, favorite_titles, selected_genres, min_year)
        st.rerun()

    if st.session_state.get("feedback_notice"):
        st.success(str(st.session_state["feedback_notice"]))
        st.session_state["feedback_notice"] = ""

    results_records = list(st.session_state.get("recommendation_results", []))
    if not results_records:
        if int(feedback_snapshot.get("total_events", 0)) > 0:
            st.markdown("### Feedback Memory")
            _render_feedback_snapshot(feedback_snapshot)
        return

    context = st.session_state.get("recommendation_context", {})
    if snapshot:
        st.success(f"Generated with {snapshot['model_label']}.")
    st.caption(
        "Built from: "
        + ", ".join(context.get("favorite_titles", []))
        + (
            f" | Genre filter: {', '.join(context.get('selected_genres', []))}"
            if context.get("selected_genres")
            else ""
        )
    )

    st.markdown("### Your picks")
    for rank, record in enumerate(results_records, start=1):
        with st.container(border=True):
            summary_left, summary_right = st.columns((0.7, 4.3), gap="large")
            with summary_left:
                st.metric(f"#{rank}", f"{float(record.get('score', 0.0)):.3f}")
                st.caption(_score_band(float(record.get("score", 0.0))))
            with summary_right:
                st.markdown(f"#### {record.get('title', 'Untitled')}")
                meta_parts = []
                release_year = record.get("release_year")
                if release_year == release_year and release_year is not None:
                    meta_parts.append(str(int(release_year)))
                if record.get("genres"):
                    meta_parts.append(str(record.get("genres")))
                mean_rating = record.get("mean_rating")
                if mean_rating == mean_rating and mean_rating is not None:
                    meta_parts.append(f"Avg rating {float(mean_rating):.2f}")
                rating_count = record.get("rating_count")
                if rating_count == rating_count and rating_count is not None:
                    meta_parts.append(f"{int(rating_count):,} ratings")
                st.caption(" | ".join(meta_parts))
                badge_markup = _badge_markup(str(record.get("explanation_tags", "")))
                if badge_markup:
                    st.markdown(badge_markup, unsafe_allow_html=True)
                st.write(str(record.get("reason", "")))
                if record.get("explanation_details"):
                    st.caption(str(record.get("explanation_details")))

                feedback_left, feedback_mid, feedback_right = st.columns(3)
                if feedback_left.button("Like", key=f"feedback_like_{record.get('movie_id')}_{rank}", use_container_width=True):
                    _submit_feedback(record, rank, "like", snapshot)
                if feedback_mid.button("Save for later", key=f"feedback_save_{record.get('movie_id')}_{rank}", use_container_width=True):
                    _submit_feedback(record, rank, "save", snapshot)
                if feedback_right.button("Not for me", key=f"feedback_not_for_me_{record.get('movie_id')}_{rank}", use_container_width=True):
                    _submit_feedback(record, rank, "not_for_me", snapshot)

    if int(feedback_snapshot.get("total_events", 0)) > 0:
        st.markdown("### Feedback Memory")
        _render_feedback_snapshot(feedback_snapshot)


def _render_eda(bundle, catalog, model_card: dict[str, Any] | None, feedback_frame) -> None:
    st.markdown("## Exploratory Data Analysis")
    st.write(
        "This section turns the project visuals into live charts so the product can "
        "show how the data behaves, not just what the final model recommends."
    )

    ratings_tab, genre_tab, release_tab, director_tab, model_tab, feedback_tab = st.tabs(
        ["Ratings", "Genres", "Release Years", "Directors", "Model Performance", "Feedback"]
    )

    with ratings_tab:
        if bundle is None or catalog is None:
            st.info("Ratings visuals become available as soon as the dataset files are loaded locally.")
        else:
            st.plotly_chart(eda.ratings_distribution(bundle.train), use_container_width=True)
            st.plotly_chart(eda.top_rated_movies(catalog), use_container_width=True)

    with genre_tab:
        if catalog is None:
            st.info("Genre visuals become available as soon as the catalog is loaded.")
        else:
            st.plotly_chart(eda.genre_distribution(catalog), use_container_width=True)

    with release_tab:
        if catalog is None:
            st.info("Release-year visuals become available as soon as the catalog is loaded.")
        else:
            st.plotly_chart(eda.movies_per_year(catalog), use_container_width=True)

    with director_tab:
        if catalog is None:
            st.info("Director-level visuals become available once the catalog is loaded.")
        else:
            director_figure = eda.top_directors(catalog)
            if director_figure is None:
                st.info("Director-level visuals will appear once `imdb_data.csv` is available.")
            else:
                st.plotly_chart(director_figure, use_container_width=True)

    with model_tab:
        _render_model_performance_panel(model_card)

    with feedback_tab:
        if feedback_frame is None or getattr(feedback_frame, "empty", True):
            st.info("User feedback charts will appear after people start reacting to recommendations.")
        else:
            _render_feedback_snapshot(feedback_summary())
            action_chart = eda.feedback_actions_chart(feedback_frame)
            title_chart = eda.feedback_top_titles_chart(feedback_frame)
            chart_left, chart_right = st.columns(2, gap="large")
            with chart_left:
                if action_chart is not None:
                    st.plotly_chart(action_chart, use_container_width=True)
            with chart_right:
                if title_chart is not None:
                    st.plotly_chart(title_chart, use_container_width=True)


def _render_about() -> None:
    st.markdown("## About ScreenLot")
    _render_banner_image()

    intro_left, intro_right = st.columns((1.2, 0.8), gap="large")
    with intro_left:
        st.markdown(
            (
                "<div class='contributor-shell'>"
                "<h3 class='section-title'>Project story</h3>"
                f"<div class='section-copy'>{ABOUT_PROJECT}</div>"
                f"<div class='section-copy' style='margin-top: 1rem;'>{MODEL_STRATEGY}</div>"
                f"<div class='small-note' style='margin-top: 1rem;'>Supervisor: {SUPERVISOR}</div>"
                "</div>"
            ),
            unsafe_allow_html=True,
        )

    with intro_right:
        dataport_path = _safe_image(DATAPORT_WORDMARK)
        if dataport_path:
            st.image(dataport_path, use_container_width=True)
        st.caption(f"Product concept and presentation material reference {ORGANIZATION}.")

    st.markdown("### Contributors")
    st.caption("The original developers are now represented directly inside the product with their images and roles.")
    row = st.columns(3, gap="large")
    for index, contributor in enumerate(CONTRIBUTORS):
        column = row[index % 3]
        with column:
            with st.container(border=True):
                image_path = _safe_image(contributor.image_path)
                if image_path:
                    st.image(image_path, use_container_width=True)
                st.markdown(f"#### {contributor.name}")
                st.caption(contributor.role)
                st.write(contributor.bio)


def _model_artifact_names() -> list[str]:
    if not MODEL_ARCHIVE.exists():
        return []
    with zipfile.ZipFile(MODEL_ARCHIVE) as archive:
        return sorted(
            name for name in archive.namelist() if name.lower().endswith(".pkl")
        )


def _render_report(
    bundle,
    catalog,
    collaborative_status: str,
    snapshot: dict[str, Any] | None,
    model_card: dict[str, Any] | None,
    feedback_snapshot: dict[str, Any],
    feedback_frame,
) -> None:
    st.markdown("## Report and Conclusion")
    st.write(
        "This section keeps the original project narrative visible while grounding the "
        "app in what is already implemented and what still needs to be trained."
    )

    metrics_left, metrics_mid, metrics_right = st.columns(3)
    if snapshot:
        serving_row = snapshot["serving_row"]
        metrics_left.metric("Serving Model", snapshot["selected_model_label"])
        metrics_mid.metric("Live Recall@20", _format_metric_value(serving_row.get("recall@20")))
        metrics_right.metric("Live NDCG@20", _format_metric_value(serving_row.get("ndcg@20")))
    else:
        metrics_left.metric("Slide Benchmark RMSE", f"{BENCHMARK_RMSE:.3f}")
        metrics_mid.metric("Archived Model Files", len(_model_artifact_names()))
        metrics_right.metric("Live App State", "Ready" if bundle is not None and catalog is not None else "Waiting for data")

    st.markdown(
        (
            "<div class='report-shell'>"
            "<h3 class='section-title'>Summary</h3>"
            f"<div class='section-copy'>{REPORT_SUMMARY}</div>"
            f"<div class='section-copy' style='margin-top: 1rem;'>Collaborative artifact status: {collaborative_status}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown("### Current product conclusion")
    st.write(
        "ScreenLot now has a live benchmarked serving model rather than just a slide-deck baseline. "
        "The next big lift is productization: capture recommendation feedback, keep retraining on the "
        "open-data pipeline, and compare user acceptance against these offline ranking metrics."
    )
    st.markdown("### Live benchmark detail")
    _render_model_performance_panel(model_card)

    st.markdown("### Recommendation feedback")
    _render_feedback_snapshot(feedback_snapshot)
    if feedback_frame is None or getattr(feedback_frame, "empty", True):
        st.info("No persisted feedback yet. Once people react to recommendations, the report will summarize it here.")
    else:
        feedback_left, feedback_right = st.columns(2, gap="large")
        with feedback_left:
            action_chart = eda.feedback_actions_chart(feedback_frame)
            if action_chart is not None:
                st.plotly_chart(action_chart, use_container_width=True)
        with feedback_right:
            title_chart = eda.feedback_top_titles_chart(feedback_frame)
            if title_chart is not None:
                st.plotly_chart(title_chart, use_container_width=True)
        st.caption(f"Feedback log path: {FEEDBACK_LOG_PATH}")

    st.markdown("### Deployment and persistence")
    st.code(
        "\n".join(
            [
                f"SCREENLOT_DATA_DIR={DEFAULT_DATA_DIR}",
                f"SCREENLOT_STATE_DIR={STATE_DIR}",
                f"SCREENLOT_FEEDBACK_LOG={FEEDBACK_LOG_PATH}",
            ]
        ),
        language="bash",
    )
    st.caption(
        "The app now supports environment-driven data, artifact, and feedback paths so the same code can move "
        "from this local repo into a hosted Streamlit deployment more cleanly."
    )


def _render_suggestions() -> None:
    st.markdown("## Suggestions")
    st.write("This page captures practical next steps for turning ScreenLot into a stronger product.")

    for item in SUGGESTION_ITEMS:
        st.markdown(f"- {item}")

    st.markdown("### Product notes")
    st.text_area(
        "What should ScreenLot learn next?",
        value=(
            "Add persistent profiles, recommendation acceptance logging, and a scheduled "
            "retraining workflow once the original competition data is staged."
        ),
        height=140,
    )

    st.markdown("### Deployment Notes")
    st.code(
        "\n".join(
            [
                f"SCREENLOT_DATA_DIR={DEFAULT_DATA_DIR}",
                f"SCREENLOT_STATE_DIR={STATE_DIR}",
                f"SCREENLOT_FEEDBACK_LOG={FEEDBACK_LOG_PATH}",
            ]
        ),
        language="bash",
    )
    st.caption(
        "ScreenLot now reads its data and persistence paths from environment-friendly runtime settings, "
        "which makes it easier to deploy outside this workstation."
    )


def main() -> None:
    _apply_page_config()

    data_dir = DEFAULT_DATA_DIR
    readiness = dataset_status(data_dir)
    missing = missing_required_files(data_dir)
    saved_model_card = _load_saved_model_card()
    snapshot = model_snapshot(saved_model_card)
    feedback_frame = load_feedback_frame()
    feedback_snapshot = feedback_summary()
    page = _render_sidebar(
        data_dir=data_dir,
        readiness=readiness,
        snapshot=snapshot,
        feedback_snapshot=feedback_snapshot,
    )

    bundle = None
    catalog = None
    engine = None
    collaborative_status = "Collaborative archive not yet loaded"
    metrics = None

    if not missing:
        try:
            bundle, catalog = _load_bundle(str(data_dir))
            metrics = dataset_metrics(bundle, catalog)
            engine, collaborative_status = _load_recommender(str(data_dir))
        except Exception as exc:  # pragma: no cover - surfaced directly in the app.
            st.warning(f"ScreenLot found the dataset folder but could not finish setup: {exc}")

    if page == "Home":
        _render_home(metrics, collaborative_status, snapshot, feedback_snapshot)
    elif page == "Recommendation Engine":
        _render_recommendations(
            catalog,
            engine,
            collaborative_status,
            snapshot,
            saved_model_card,
            feedback_snapshot,
        )
    elif page == "EDA":
        _render_eda(bundle, catalog, saved_model_card, feedback_frame)
    elif page == "About":
        _render_about()
    elif page == "Report and Conclusion":
        _render_report(
            bundle,
            catalog,
            collaborative_status,
            snapshot,
            saved_model_card,
            feedback_snapshot,
            feedback_frame,
        )
    elif page == "Suggestions":
        _render_suggestions()
