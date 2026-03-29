from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .runtime import DEFAULT_DATA_DIR


PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
BRANDING_DIR = ASSETS_DIR / "branding"
TEAM_DIR = ASSETS_DIR / "team"
PREVIEWS_DIR = ASSETS_DIR / "previews"
MODEL_ARCHIVE = PROJECT_ROOT / "Model pickles.zip"

SCREENLOT_LOGO = BRANDING_DIR / "screenlot-logo.png"
SCREENLOT_BANNER = BRANDING_DIR / "screenlot-banner.svg"
DATAPORT_WORDMARK = BRANDING_DIR / "dataport-wordmark.png"
STREAMLIT_CONCEPT = PREVIEWS_DIR / "streamlit-concept.png"

APP_NAME = "ScreenLot"
APP_TAGLINE = (
    "A cinematic discovery experience that blends explainable recommendations, live "
    "analytics, and project storytelling into one modern streaming-style product."
)
SUPERVISOR = "Nomfundo Manyisa"
ORGANIZATION = "Data Port Incorporated"
BENCHMARK_RMSE = 0.786


@dataclass(frozen=True)
class Contributor:
    name: str
    role: str
    bio: str
    image_path: Path


CONTRIBUTORS = (
    Contributor(
        name="Prince Okon",
        role="Team Lead",
        bio=(
            "Led the project direction, coordinated the modeling workflow, and drove "
            "the case for turning the recommendation notebook into a usable product."
        ),
        image_path=TEAM_DIR / "prince-okon.png",
    ),
    Contributor(
        name="Daniel Odukoya",
        role="Administrator",
        bio=(
            "Supported project administration and helped keep the group aligned on "
            "delivery, documentation, and execution."
        ),
        image_path=TEAM_DIR / "daniel-odukoya.png",
    ),
    Contributor(
        name="Huzaifa Abu",
        role="Technical Lead",
        bio=(
            "Focused on technical implementation, experimentation, and making the "
            "recommendation workflow more robust."
        ),
        image_path=TEAM_DIR / "huzaifa-abu.png",
    ),
    Contributor(
        name="Jerry Iriri",
        role="Chief Designer",
        bio=(
            "Helped shape the visual direction of the project and the presentation of "
            "the ScreenLot product idea."
        ),
        image_path=TEAM_DIR / "jerry-iriri.png",
    ),
    Contributor(
        name="Izunna Eneude",
        role="Quality Control",
        bio=(
            "Focused on quality control, review, and keeping the project story and "
            "deliverables coherent."
        ),
        image_path=TEAM_DIR / "izunna-eneude.png",
    ),
)

NAVIGATION = (
    "Home",
    "Recommendation Engine",
    "EDA",
    "About",
    "Report and Conclusion",
    "Suggestions",
)

ABOUT_PROJECT = (
    "ScreenLot is the product layer for the original movie recommendation project. "
    "The app turns a notebook-led unsupervised learning exercise into a polished "
    "decision-support experience for discovery, exploration, and communication."
)

ECONOMIC_VALUE = (
    "Recommendation systems improve platform affinity, reduce decision fatigue, and "
    "help users spend more time on relevant content. For streaming businesses, that "
    "translates into stronger retention, more sessions per user, and more profitable "
    "content surfacing."
)

MODEL_STRATEGY = (
    "The current product foundation uses a validation-tuned hybrid recommendation "
    "approach. Collaborative signals, metadata similarity, and popularity context are "
    "benchmarked offline, then the best-performing serving engine is surfaced directly "
    "inside the app together with its evaluation story."
)

FEATURE_PILLARS = (
    "Modern explainable recommendations from favorite titles",
    "Live benchmark and feedback visibility inside the product",
    "Explorable EDA views for ratings, releases, genres, and directors",
    "Contributor, project story, and deployment-ready product narrative",
)

REPORT_SUMMARY = (
    "The original project explored collaborative filtering models such as SVD, "
    "NormalPredictor, BaselineOnly, NMF, SlopeOne, and CoClustering. The slide deck "
    "reports a benchmark RMSE of 0.786. This app reorganizes that work into a "
    "maintainable product structure so the model and storytelling can evolve together."
)

SUGGESTION_ITEMS = (
    "Promote the current hybrid engine into a fully trained SVD++ or ranking pipeline once the project data is staged locally.",
    "Add persistent user profiles so the app can learn from explicit ratings, skips, and watchlist saves.",
    "Introduce A/B testing for recommendation explanations, onboarding prompts, and home page ordering.",
    "Track business metrics such as click-through rate, watch conversion, repeat visits, and recommendation acceptance.",
    "Create a lightweight admin console for retraining schedules, data health checks, and artifact versioning.",
)
