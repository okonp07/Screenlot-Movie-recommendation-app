from __future__ import annotations

from typing import Any


DEFAULT_THEME_MODE = "dark"

_THEME_TOKENS: dict[str, dict[str, Any]] = {
    "dark": {
        "color_scheme": "dark",
        "screenlot_ink": "#f5efff",
        "screenlot_muted": "#b9afc9",
        "screenlot_purple_ash": "#8f82a8",
        "screenlot_purple_soft": "#b2a6c7",
        "screenlot_purple_deep": "#5c4f73",
        "app_glow_primary": "rgba(143, 130, 168, 0.25)",
        "app_glow_secondary": "rgba(178, 166, 199, 0.13)",
        "app_bg_start": "#050409",
        "app_bg_mid": "#09070f",
        "app_bg_end": "#0d0914",
        "sidebar_start": "rgba(11, 9, 17, 0.98)",
        "sidebar_end": "rgba(15, 12, 24, 0.96)",
        "panel": "rgba(15, 12, 24, 0.86)",
        "panel_strong": "rgba(8, 6, 12, 0.95)",
        "edge": "rgba(178, 166, 199, 0.22)",
        "glow": "rgba(143, 130, 168, 0.18)",
        "surface_soft": "rgba(255, 255, 255, 0.035)",
        "surface_strong": "rgba(255, 255, 255, 0.015)",
        "surface_border": "rgba(255, 255, 255, 0.07)",
        "button_start": "rgba(143, 130, 168, 0.26)",
        "button_end": "rgba(92, 79, 115, 0.28)",
        "button_hover_start": "rgba(160, 148, 186, 0.32)",
        "button_hover_end": "rgba(109, 96, 133, 0.34)",
        "button_text": "#f5efff",
        "input_bg": "rgba(255, 255, 255, 0.03)",
        "input_border": "rgba(178, 166, 199, 0.16)",
        "tab_bg": "rgba(255, 255, 255, 0.03)",
        "tab_border": "rgba(178, 166, 199, 0.12)",
        "tab_active_bg": "rgba(143, 130, 168, 0.18)",
        "tab_active_border": "rgba(178, 166, 199, 0.32)",
        "callout_start": "rgba(143, 130, 168, 0.14)",
        "callout_end": "rgba(92, 79, 115, 0.08)",
        "card_shadow": "0 25px 70px rgba(0, 0, 0, 0.34)",
        "metric_shadow": "0 18px 40px rgba(0, 0, 0, 0.18)",
        "chart_paper_bg": "rgba(0, 0, 0, 0)",
        "chart_plot_bg": "rgba(17, 15, 24, 0.72)",
        "chart_grid_color": "rgba(178, 166, 199, 0.16)",
        "chart_legend_bg": "rgba(17, 15, 24, 0.45)",
        "chart_font_color": "#f5efff",
        "table_shell_bg": "rgba(10, 8, 14, 0.48)",
        "table_header_bg": "rgba(17, 15, 24, 0.86)",
        "table_row_bg": "rgba(8, 6, 12, 0.74)",
        "table_row_alt_bg": "rgba(13, 10, 19, 0.66)",
        "table_text": "#f5efff",
        "table_border": "rgba(178, 166, 199, 0.18)",
        "wordmark_filter": "none",
    },
    "light": {
        "color_scheme": "light",
        "screenlot_ink": "#221b2c",
        "screenlot_muted": "#6f677c",
        "screenlot_purple_ash": "#7b6c96",
        "screenlot_purple_soft": "#9687b1",
        "screenlot_purple_deep": "#58486f",
        "app_glow_primary": "rgba(150, 135, 177, 0.20)",
        "app_glow_secondary": "rgba(123, 108, 150, 0.14)",
        "app_bg_start": "#fcfaff",
        "app_bg_mid": "#f6f1fb",
        "app_bg_end": "#efe7f7",
        "sidebar_start": "rgba(255, 255, 255, 0.96)",
        "sidebar_end": "rgba(246, 240, 252, 0.98)",
        "panel": "rgba(255, 255, 255, 0.88)",
        "panel_strong": "rgba(248, 243, 252, 0.98)",
        "edge": "rgba(123, 108, 150, 0.18)",
        "glow": "rgba(150, 135, 177, 0.18)",
        "surface_soft": "rgba(255, 255, 255, 0.82)",
        "surface_strong": "rgba(248, 243, 252, 0.92)",
        "surface_border": "rgba(123, 108, 150, 0.12)",
        "button_start": "rgba(150, 135, 177, 0.22)",
        "button_end": "rgba(88, 72, 111, 0.16)",
        "button_hover_start": "rgba(150, 135, 177, 0.30)",
        "button_hover_end": "rgba(88, 72, 111, 0.22)",
        "button_text": "#221b2c",
        "input_bg": "rgba(255, 255, 255, 0.92)",
        "input_border": "rgba(123, 108, 150, 0.15)",
        "tab_bg": "rgba(255, 255, 255, 0.72)",
        "tab_border": "rgba(123, 108, 150, 0.10)",
        "tab_active_bg": "rgba(150, 135, 177, 0.16)",
        "tab_active_border": "rgba(123, 108, 150, 0.26)",
        "callout_start": "rgba(150, 135, 177, 0.12)",
        "callout_end": "rgba(88, 72, 111, 0.06)",
        "card_shadow": "0 22px 48px rgba(80, 55, 110, 0.10)",
        "metric_shadow": "0 14px 30px rgba(80, 55, 110, 0.08)",
        "chart_paper_bg": "rgba(255, 255, 255, 0)",
        "chart_plot_bg": "rgba(255, 255, 255, 0)",
        "chart_grid_color": "rgba(123, 108, 150, 0.12)",
        "chart_legend_bg": "rgba(255, 255, 255, 0)",
        "chart_font_color": "#000000",
        "table_shell_bg": "rgba(255, 255, 255, 0)",
        "table_header_bg": "rgba(255, 255, 255, 0)",
        "table_row_bg": "rgba(255, 255, 255, 0)",
        "table_row_alt_bg": "rgba(255, 255, 255, 0)",
        "table_text": "#000000",
        "table_border": "rgba(123, 108, 150, 0.16)",
        "wordmark_filter": "brightness(0) saturate(100%) opacity(0.92)",
    },
}


def normalize_theme_mode(mode: str | None) -> str:
    normalized = (mode or DEFAULT_THEME_MODE).strip().lower()
    if normalized not in _THEME_TOKENS:
        return DEFAULT_THEME_MODE
    return normalized


def plotly_template(mode: str | None) -> str:
    return "plotly_white" if normalize_theme_mode(mode) == "light" else "plotly_dark"


def plotly_layout(mode: str | None) -> dict[str, Any]:
    tokens = _THEME_TOKENS[normalize_theme_mode(mode)]
    return {
        "template": plotly_template(mode),
        "paper_bgcolor": tokens["chart_paper_bg"],
        "plot_bgcolor": tokens["chart_plot_bg"],
        "font": {"color": tokens["chart_font_color"]},
        "title": {"font": {"color": tokens["chart_font_color"]}},
        "legend": {
            "bgcolor": tokens["chart_legend_bg"],
            "font": {"color": tokens["chart_font_color"]},
        },
        "xaxis": {
            "gridcolor": tokens["chart_grid_color"],
            "zerolinecolor": tokens["chart_grid_color"],
            "tickfont": {"color": tokens["chart_font_color"]},
            "title": {"font": {"color": tokens["chart_font_color"]}},
        },
        "yaxis": {
            "gridcolor": tokens["chart_grid_color"],
            "zerolinecolor": tokens["chart_grid_color"],
            "tickfont": {"color": tokens["chart_font_color"]},
            "title": {"font": {"color": tokens["chart_font_color"]}},
        },
    }


def build_global_css(mode: str | None = None) -> str:
    tokens = _THEME_TOKENS[normalize_theme_mode(mode)]
    css = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    color-scheme: __color_scheme__;
    --screenlot-ink: __screenlot_ink__;
    --screenlot-muted: __screenlot_muted__;
    --screenlot-purple-ash: __screenlot_purple_ash__;
    --screenlot-purple-soft: __screenlot_purple_soft__;
    --screenlot-purple-deep: __screenlot_purple_deep__;
    --screenlot-app-glow-primary: __app_glow_primary__;
    --screenlot-app-glow-secondary: __app_glow_secondary__;
    --screenlot-app-bg-start: __app_bg_start__;
    --screenlot-app-bg-mid: __app_bg_mid__;
    --screenlot-app-bg-end: __app_bg_end__;
    --screenlot-sidebar-start: __sidebar_start__;
    --screenlot-sidebar-end: __sidebar_end__;
    --screenlot-panel: __panel__;
    --screenlot-panel-strong: __panel_strong__;
    --screenlot-edge: __edge__;
    --screenlot-glow: __glow__;
    --screenlot-surface-soft: __surface_soft__;
    --screenlot-surface-strong: __surface_strong__;
    --screenlot-surface-border: __surface_border__;
    --screenlot-button-start: __button_start__;
    --screenlot-button-end: __button_end__;
    --screenlot-button-hover-start: __button_hover_start__;
    --screenlot-button-hover-end: __button_hover_end__;
    --screenlot-button-text: __button_text__;
    --screenlot-input-bg: __input_bg__;
    --screenlot-input-border: __input_border__;
    --screenlot-tab-bg: __tab_bg__;
    --screenlot-tab-border: __tab_border__;
    --screenlot-tab-active-bg: __tab_active_bg__;
    --screenlot-tab-active-border: __tab_active_border__;
    --screenlot-callout-start: __callout_start__;
    --screenlot-callout-end: __callout_end__;
    --screenlot-card-shadow: __card_shadow__;
    --screenlot-metric-shadow: __metric_shadow__;
    --screenlot-table-shell-bg: __table_shell_bg__;
    --screenlot-table-header-bg: __table_header_bg__;
    --screenlot-table-row-bg: __table_row_bg__;
    --screenlot-table-row-alt-bg: __table_row_alt_bg__;
    --screenlot-table-text: __table_text__;
    --screenlot-table-border: __table_border__;
    --screenlot-wordmark-filter: __wordmark_filter__;
}

html, body, [class*="st-"], [data-testid="stMarkdownContainer"] {
    font-family: "Manrope", "Avenir Next", "Segoe UI", sans-serif;
}

h1, h2, h3, h4, .hero-title, .banner-title {
    font-family: "Cormorant Garamond", "Georgia", serif;
    letter-spacing: 0.02em;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 12% 0%, var(--screenlot-app-glow-primary), transparent 26%),
        radial-gradient(circle at 85% 12%, var(--screenlot-app-glow-secondary), transparent 22%),
        linear-gradient(
            180deg,
            var(--screenlot-app-bg-start) 0%,
            var(--screenlot-app-bg-mid) 46%,
            var(--screenlot-app-bg-end) 100%
        );
    color: var(--screenlot-ink);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, var(--screenlot-sidebar-start) 0%, var(--screenlot-sidebar-end) 100%);
    border-right: 1px solid var(--screenlot-edge);
}

[data-testid="stSidebar"] > div:first-child {
    backdrop-filter: blur(18px);
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
}

[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] li,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] li,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] div {
    color: var(--screenlot-ink);
}

[data-testid="stCaptionContainer"] p,
[data-testid="stCaptionContainer"] span,
.hero-copy,
.section-copy,
.small-note,
.contributor-meta {
    color: var(--screenlot-muted) !important;
    line-height: 1.75;
}

.hero-shell,
.section-shell,
.report-shell,
.status-shell,
.banner-shell,
.contributor-shell,
.insight-shell {
    position: relative;
    overflow: hidden;
    background: linear-gradient(180deg, var(--screenlot-panel) 0%, var(--screenlot-panel-strong) 100%);
    border: 1px solid var(--screenlot-edge);
    border-radius: 28px;
    padding: 1.35rem 1.45rem;
    box-shadow:
        var(--screenlot-card-shadow),
        inset 0 1px 0 rgba(255, 255, 255, 0.04);
}

.hero-shell::before,
.section-shell::before,
.report-shell::before,
.banner-shell::before,
.contributor-shell::before,
.insight-shell::before {
    content: "";
    position: absolute;
    inset: -20% auto auto -10%;
    width: 220px;
    height: 220px;
    background: radial-gradient(circle, var(--screenlot-glow) 0%, transparent 68%);
    pointer-events: none;
}

.banner-shell {
    padding: 0.6rem;
}

.hero-shell {
    padding: 2rem;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.8rem;
}

.brand-mark {
    width: 72px;
    height: 72px;
    border-radius: 22px;
    object-fit: contain;
    background: var(--screenlot-surface-soft);
    border: 1px solid var(--screenlot-edge);
    padding: 0.55rem;
}

.kicker {
    color: var(--screenlot-purple-soft);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}

.hero-title {
    font-size: 3.2rem;
    line-height: 0.96;
    margin: 0.3rem 0 0.85rem 0;
}

.pill-strip,
.explanation-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.55rem;
}

.pill-strip {
    margin-top: 1rem;
}

.pill,
.explanation-chip {
    border-radius: 999px;
    color: var(--screenlot-ink);
    font-size: 0.82rem;
    padding: 0.35rem 0.78rem;
    border: 1px solid var(--screenlot-edge);
}

.pill {
    background: linear-gradient(180deg, var(--screenlot-button-start) 0%, var(--screenlot-button-end) 100%);
}

.explanation-strip {
    margin: 0.55rem 0 0.8rem 0;
}

.explanation-chip {
    background: var(--screenlot-surface-soft);
    font-weight: 700;
}

.section-title {
    margin-top: 0;
    margin-bottom: 0.35rem;
}

.metric-strip {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.9rem;
}

.metric-card {
    background: linear-gradient(180deg, var(--screenlot-surface-soft) 0%, var(--screenlot-surface-strong) 100%);
    border: 1px solid var(--screenlot-surface-border);
    border-radius: 22px;
    padding: 1rem 1.05rem;
    box-shadow: var(--screenlot-metric-shadow);
}

.metric-label {
    color: var(--screenlot-muted);
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.16em;
}

.metric-value {
    color: var(--screenlot-ink);
    font-size: 1.65rem;
    font-weight: 800;
    margin-top: 0.15rem;
}

.screenlot-table-shell {
    overflow-x: auto;
    background: var(--screenlot-table-shell-bg);
    border: 1px solid var(--screenlot-table-border);
    border-radius: 22px;
}

.screenlot-table {
    width: 100%;
    border-collapse: collapse;
    background: transparent;
    color: var(--screenlot-table-text);
}

.screenlot-table thead tr {
    background: var(--screenlot-table-header-bg);
}

.screenlot-table tbody tr {
    background: var(--screenlot-table-row-bg);
}

.screenlot-table tbody tr:nth-child(even) {
    background: var(--screenlot-table-row-alt-bg);
}

.screenlot-table th,
.screenlot-table td {
    padding: 1rem 0.95rem;
    text-align: left;
    color: var(--screenlot-table-text);
    border-right: 1px solid var(--screenlot-table-border);
    border-bottom: 1px solid var(--screenlot-table-border);
    white-space: nowrap;
}

.screenlot-table th:last-child,
.screenlot-table td:last-child {
    border-right: none;
}

.screenlot-table tbody tr:last-child td {
    border-bottom: none;
}

.screenlot-wordmark-image {
    display: block;
    width: 100%;
    height: auto;
    filter: var(--screenlot-wordmark-filter);
}

.screenlot-banner-shell {
    margin-bottom: 1.25rem;
}

.screenlot-banner-image {
    display: block;
    width: 100%;
    height: auto;
    border-radius: 24px;
}

.contributor-card {
    background: linear-gradient(180deg, var(--screenlot-surface-soft) 0%, var(--screenlot-surface-strong) 100%);
    border: 1px solid var(--screenlot-edge);
    border-radius: 24px;
    padding: 1rem;
    min-height: 100%;
}

.contributor-role {
    color: var(--screenlot-purple-soft);
    font-size: 0.84rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 800;
}

.banner-title {
    font-size: 2.8rem;
    line-height: 0.95;
}

.insight-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
}

.status-callout {
    background: linear-gradient(180deg, var(--screenlot-callout-start) 0%, var(--screenlot-callout-end) 100%);
    border: 1px solid var(--screenlot-edge);
    border-radius: 22px;
    padding: 1rem 1.1rem;
}

[data-testid="stButton"] > button,
[data-testid="baseButton-secondary"] {
    border-radius: 16px;
    border: 1px solid var(--screenlot-edge);
    background: linear-gradient(180deg, var(--screenlot-button-start) 0%, var(--screenlot-button-end) 100%);
    color: var(--screenlot-button-text);
    font-weight: 700;
}

[data-testid="stButton"] > button:hover {
    border-color: var(--screenlot-purple-soft);
    background: linear-gradient(
        180deg,
        var(--screenlot-button-hover-start) 0%,
        var(--screenlot-button-hover-end) 100%
    );
}

[data-baseweb="select"] > div,
[data-testid="stTextInputRootElement"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: var(--screenlot-input-bg) !important;
    border: 1px solid var(--screenlot-input-border) !important;
    border-radius: 16px !important;
    color: var(--screenlot-ink) !important;
}

[data-baseweb="select"] * {
    color: var(--screenlot-ink) !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, var(--screenlot-surface-soft) 0%, var(--screenlot-surface-strong) 100%);
    border: 1px solid var(--screenlot-edge);
    border-radius: 20px;
    padding: 0.75rem 1rem;
    box-shadow: var(--screenlot-metric-shadow);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricDelta"] {
    color: var(--screenlot-muted) !important;
}

[data-testid="stMetricValue"] {
    color: var(--screenlot-ink) !important;
}

div[data-baseweb="tab-list"] {
    gap: 0.35rem;
}

button[role="tab"] {
    border-radius: 999px !important;
    background: var(--screenlot-tab-bg) !important;
    border: 1px solid var(--screenlot-tab-border) !important;
    color: var(--screenlot-ink) !important;
}

button[role="tab"][aria-selected="true"] {
    background: var(--screenlot-tab-active-bg) !important;
    border-color: var(--screenlot-tab-active-border) !important;
}

@media (max-width: 900px) {
    .hero-title {
        font-size: 2.4rem;
    }

    .metric-strip,
    .insight-grid {
        grid-template-columns: 1fr;
    }
}
"""
    for key, value in tokens.items():
        css = css.replace(f"__{key}__", str(value))
    return css


GLOBAL_CSS = build_global_css(DEFAULT_THEME_MODE)
