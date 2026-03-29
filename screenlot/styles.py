GLOBAL_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --screenlot-ink: #f5efff;
    --screenlot-muted: #b9afc9;
    --screenlot-purple-ash: #8f82a8;
    --screenlot-purple-soft: #b2a6c7;
    --screenlot-purple-deep: #5c4f73;
    --screenlot-black: #07060c;
    --screenlot-black-soft: #110f18;
    --screenlot-panel: rgba(15, 12, 24, 0.86);
    --screenlot-panel-strong: rgba(8, 6, 12, 0.95);
    --screenlot-edge: rgba(178, 166, 199, 0.22);
    --screenlot-glow: rgba(143, 130, 168, 0.18);
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
        radial-gradient(circle at 12% 0%, rgba(143, 130, 168, 0.25), transparent 26%),
        radial-gradient(circle at 85% 12%, rgba(178, 166, 199, 0.13), transparent 22%),
        linear-gradient(180deg, #050409 0%, #09070f 46%, #0d0914 100%);
    color: var(--screenlot-ink);
}

[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, rgba(11, 9, 17, 0.98) 0%, rgba(15, 12, 24, 0.96) 100%);
    border-right: 1px solid var(--screenlot-edge);
}

[data-testid="stHeader"] {
    background: rgba(0, 0, 0, 0);
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
        0 25px 70px rgba(0, 0, 0, 0.34),
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
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(178, 166, 199, 0.18);
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

.hero-copy,
.section-copy,
.small-note,
.contributor-meta {
    color: var(--screenlot-muted);
    line-height: 1.75;
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
    border: 1px solid rgba(178, 166, 199, 0.22);
}

.pill {
    background: linear-gradient(180deg, rgba(143, 130, 168, 0.18) 0%, rgba(92, 79, 115, 0.15) 100%);
}

.explanation-strip {
    margin: 0.55rem 0 0.8rem 0;
}

.explanation-chip {
    background: rgba(178, 166, 199, 0.1);
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
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.035) 0%, rgba(255, 255, 255, 0.015) 100%);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 22px;
    padding: 1rem 1.05rem;
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

.contributor-card {
    background: linear-gradient(180deg, rgba(255, 255, 255, 0.03) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(178, 166, 199, 0.18);
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
    background: linear-gradient(180deg, rgba(143, 130, 168, 0.14) 0%, rgba(92, 79, 115, 0.08) 100%);
    border: 1px solid rgba(178, 166, 199, 0.2);
    border-radius: 22px;
    padding: 1rem 1.1rem;
}

[data-testid="stButton"] > button,
[data-testid="baseButton-secondary"] {
    border-radius: 16px;
    border: 1px solid rgba(178, 166, 199, 0.22);
    background: linear-gradient(180deg, rgba(143, 130, 168, 0.26) 0%, rgba(92, 79, 115, 0.28) 100%);
    color: var(--screenlot-ink);
    font-weight: 700;
}

[data-testid="stButton"] > button:hover {
    border-color: rgba(178, 166, 199, 0.38);
    background: linear-gradient(180deg, rgba(160, 148, 186, 0.32) 0%, rgba(109, 96, 133, 0.34) 100%);
}

[data-baseweb="select"] > div,
[data-testid="stTextInputRootElement"] > div,
[data-testid="stNumberInput"] input,
[data-testid="stTextArea"] textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(178, 166, 199, 0.16) !important;
    border-radius: 16px !important;
}

[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(178, 166, 199, 0.15);
    border-radius: 20px;
    padding: 0.75rem 1rem;
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
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(178, 166, 199, 0.12) !important;
}

button[role="tab"][aria-selected="true"] {
    background: rgba(143, 130, 168, 0.18) !important;
    border-color: rgba(178, 166, 199, 0.32) !important;
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
