import os

# Shared UI palette for Plotly (import where charts are built)
CHART_SURFACE = "#0f1419"
ACCENT = "#14b8a6"
ACCENT_FILL_A = "rgba(20, 184, 166, 0.14)"
ACCENT_FILL_B = "rgba(20, 184, 166, 0.26)"
ACCENT_FILL_C = "rgba(20, 184, 166, 0.42)"
GOAL_LINE = "#f87171"
MACRO_PIE_COLORS = ["#14b8a6", "#6366f1", "#e879a8"]

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
FAVICON_PATH = os.path.join(_APP_DIR, "assets", "favicon.svg")


def render_page_header(st, title, subtitle, *, kicker="", show_wordmark=True):
    """Consistent page title area: optional Calorify AI wordmark, kicker, title, subtitle."""
    brand_html = ""
    if show_wordmark:
        brand_html = """<div class="brand-row" aria-label="Calorify AI">
<span class="brand-wordmark">Calorify</span><span class="brand-ai">AI</span>
</div>"""
    kicker_html = f'<div class="page-header-kicker">{kicker}</div>' if kicker else ""
    st.markdown(
        f"""<div class="page-header">
{brand_html}
{kicker_html}
<h1 class="page-header-title">{title}</h1>
<p class="page-header-subtitle">{subtitle}</p>
</div>""",
        unsafe_allow_html=True,
    )


def apply_glass_style(st):
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(165deg, #0a0f18 0%, #0f172a 42%, #0c1019 100%);
        }

        .main .block-container {
            padding-top: 1.75rem;
            padding-bottom: 2.5rem;
        }

        h1, h2, h3 {
            color: #f1f5f9 !important;
            font-weight: 600 !important;
            letter-spacing: -0.02em;
        }

        .page-header {
            margin-bottom: 1.5rem;
            padding-bottom: 1.15rem;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }

        .brand-row {
            display: flex;
            align-items: baseline;
            gap: 0.35rem;
            margin-bottom: 0.55rem;
        }

        .brand-wordmark {
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.14em;
            text-transform: uppercase;
            background: linear-gradient(105deg, #5eead4 0%, #14b8a6 55%, #0d9488 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
        }

        .brand-ai {
            font-size: 0.68rem;
            font-weight: 600;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            color: #64748b;
        }

        .page-header-kicker {
            font-size: 0.7rem;
            font-weight: 600;
            letter-spacing: 0.1em;
            text-transform: uppercase;
            color: #2dd4bf;
            margin-bottom: 0.4rem;
        }

        .page-header-title {
            font-size: 1.8rem;
            font-weight: 700;
            color: #f8fafc;
            letter-spacing: -0.03em;
            line-height: 1.2;
            margin: 0;
        }

        .page-header-subtitle {
            color: #94a3b8;
            font-size: 0.95rem;
            margin: 0.45rem 0 0 0;
            line-height: 1.55;
            font-weight: 400;
            max-width: 48rem;
        }

        .glass-card, .glass-box {
            background: rgba(30, 41, 59, 0.42);
            border: 1px solid rgba(148, 163, 184, 0.12);
            backdrop-filter: blur(18px);
            -webkit-backdrop-filter: blur(18px);
            border-radius: 14px;
            padding: 1.1rem 1.3rem;
            box-shadow:
                0 1px 2px rgba(0, 0, 0, 0.18),
                0 12px 28px -8px rgba(0, 0, 0, 0.45);
            margin-bottom: 1rem;
        }

        div[data-testid="stMetric"] {
            background: rgba(15, 23, 42, 0.55);
            border: 1px solid rgba(148, 163, 184, 0.1);
            border-radius: 12px;
            padding: 14px 16px;
        }

        div[data-testid="stMetric"] label {
            color: #94a3b8 !important;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.96) 0%, rgba(10, 15, 28, 0.98) 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.1);
        }

        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.25rem;
        }

        .stButton > button {
            width: 100%;
            border-radius: 10px;
            border: 1px solid rgba(20, 184, 166, 0.35);
            background: linear-gradient(180deg, rgba(20, 184, 166, 0.22) 0%, rgba(20, 184, 166, 0.06) 100%);
            color: #ecfdf5;
            font-weight: 600;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        .stButton > button:hover {
            border-color: rgba(45, 212, 191, 0.55);
            box-shadow: 0 0 0 1px rgba(20, 184, 166, 0.15), 0 8px 22px rgba(0, 0, 0, 0.28);
        }

        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            border-radius: 10px !important;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 6px;
            background-color: rgba(15, 23, 42, 0.55);
            padding: 6px;
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
        }

        .stTabs [aria-selected="true"] {
            background: rgba(20, 184, 166, 0.14) !important;
            color: #f0fdfa !important;
        }

        div[data-testid="stFileUploader"] {
            border-radius: 12px;
            border: 1px dashed rgba(148, 163, 184, 0.28) !important;
            background: rgba(15, 23, 42, 0.35);
            padding: 0.45rem;
        }

        div[data-testid="stDataFrame"] {
            border-radius: 12px;
            border: 1px solid rgba(148, 163, 184, 0.12);
            overflow: hidden;
        }

        hr {
            border-color: rgba(148, 163, 184, 0.12) !important;
        }

        div[data-testid="stProgress"] > div > div > div {
            background-color: #14b8a6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
