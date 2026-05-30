"""
styles.py — Premium Dark Theme for V5 Fraud Intelligence Dashboard
═══════════════════════════════════════════════════════════════════
Glassmorphism cards, gradient accents, smooth animations.
"""


def get_main_css():
    """Returns the main CSS theme for the dashboard."""
    return """
    <style>
    /* ── Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-primary: #0a0e1a;
        --bg-secondary: #111827;
        --bg-card: rgba(17, 24, 39, 0.7);
        --bg-card-hover: rgba(17, 24, 39, 0.9);
        --border-glass: rgba(255, 255, 255, 0.08);
        --border-glow: rgba(99, 102, 241, 0.3);

        --text-primary: #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted: #64748b;

        --accent-blue: #6366f1;
        --accent-purple: #8b5cf6;
        --accent-cyan: #06b6d4;
        --accent-gradient: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);

        --danger: #ef4444;
        --danger-glow: rgba(239, 68, 68, 0.3);
        --warning-orange: #f97316;
        --warning-orange-glow: rgba(249, 115, 22, 0.3);
        --warning-amber: #f59e0b;
        --warning-amber-glow: rgba(245, 158, 11, 0.3);
        --success: #10b981;
        --success-glow: rgba(16, 185, 129, 0.3);

        --font-main: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        --font-mono: 'JetBrains Mono', 'Fira Code', monospace;

        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 16px;
        --radius-xl: 20px;

        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.3);
        --shadow-md: 0 4px 16px rgba(0, 0, 0, 0.4);
        --shadow-lg: 0 8px 32px rgba(0, 0, 0, 0.5);
        --shadow-glow: 0 0 20px rgba(99, 102, 241, 0.15);
    }

    /* ── Global Reset ── */
    .stApp {
        background: var(--bg-primary) !important;
        font-family: var(--font-main) !important;
        color: var(--text-primary) !important;
    }

    .stApp > header {
        background: transparent !important;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e1b4b 100%) !important;
        border-right: 1px solid var(--border-glass) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: var(--text-primary) !important;
        font-weight: 700 !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-secondary) !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: var(--text-primary) !important;
    }

    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-secondary) !important;
    }

    /* ── Typography ── */
    h1, h2, h3 {
        font-family: var(--font-main) !important;
        color: var(--text-primary) !important;
        letter-spacing: -0.03em;
    }

    h1 { font-weight: 800 !important; }
    h2 { font-weight: 700 !important; }
    h3 { font-weight: 600 !important; }

    p, span, div {
        font-family: var(--font-main) !important;
    }

    /* ── Glassmorphism Card ── */
    .glass-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 24px;
        box-shadow: var(--shadow-md);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .glass-card:hover {
        background: var(--bg-card-hover);
        border-color: var(--border-glow);
        box-shadow: var(--shadow-glow);
        transform: translateY(-2px);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: var(--bg-card);
        backdrop-filter: blur(20px);
        border: 1px solid var(--border-glass);
        border-radius: var(--radius-lg);
        padding: 20px 24px;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    }

    .metric-card.total::before { background: var(--accent-gradient); }
    .metric-card.danger::before { background: var(--danger); }
    .metric-card.warning::before { background: var(--warning-orange); }
    .metric-card.review::before { background: var(--warning-amber); }
    .metric-card.success::before { background: var(--success); }

    .metric-card:hover {
        border-color: var(--border-glow);
        transform: translateY(-3px);
        box-shadow: var(--shadow-lg);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 8px 0 4px;
        line-height: 1;
    }

    .metric-label {
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--text-muted);
    }

    .metric-delta {
        font-size: 0.75rem;
        font-weight: 600;
        margin-top: 4px;
    }

    .metric-card.total .metric-value { color: var(--accent-blue); }
    .metric-card.danger .metric-value { color: var(--danger); }
    .metric-card.warning .metric-value { color: var(--warning-orange); }
    .metric-card.review .metric-value { color: var(--warning-amber); }
    .metric-card.success .metric-value { color: var(--success); }

    /* ── Decision Badges ── */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        border: 1px solid transparent;
    }

    .badge-block {
        background: rgba(239, 68, 68, 0.15);
        color: #fca5a5;
        border-color: rgba(239, 68, 68, 0.3);
        box-shadow: 0 0 12px rgba(239, 68, 68, 0.2);
    }

    .badge-novel {
        background: rgba(249, 115, 22, 0.15);
        color: #fdba74;
        border-color: rgba(249, 115, 22, 0.3);
        box-shadow: 0 0 12px rgba(249, 115, 22, 0.2);
    }

    .badge-review {
        background: rgba(245, 158, 11, 0.15);
        color: #fcd34d;
        border-color: rgba(245, 158, 11, 0.3);
        box-shadow: 0 0 12px rgba(245, 158, 11, 0.2);
    }

    .badge-allow {
        background: rgba(16, 185, 129, 0.15);
        color: #6ee7b7;
        border-color: rgba(16, 185, 129, 0.3);
    }

    /* ── Buttons ── */
    .stButton > button {
        background: var(--accent-gradient) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-md) !important;
        padding: 12px 32px !important;
        font-weight: 600 !important;
        font-family: var(--font-main) !important;
        letter-spacing: 0.02em;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3) !important;
    }

    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.45) !important;
    }

    .stButton > button:active {
        transform: translateY(0) !important;
    }

    /* ── File Uploader ── */
    .stFileUploader {
        border: 2px dashed var(--border-glow) !important;
        border-radius: var(--radius-lg) !important;
        background: rgba(99, 102, 241, 0.05) !important;
        transition: all 0.3s ease;
    }

    .stFileUploader:hover {
        border-color: var(--accent-blue) !important;
        background: rgba(99, 102, 241, 0.1) !important;
    }

    /* ── DataFrames / Tables ── */
    .stDataFrame {
        border-radius: var(--radius-md) !important;
        overflow: hidden;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        background: var(--bg-card);
        border-radius: var(--radius-md);
        padding: 4px;
        gap: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        font-weight: 600;
        color: var(--text-secondary);
    }

    .stTabs [aria-selected="true"] {
        background: var(--accent-gradient) !important;
        color: white !important;
    }

    /* ── Progress Bar ── */
    .stProgress > div > div {
        background: var(--accent-gradient) !important;
        border-radius: 10px;
    }

    /* ── Animations ── */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes pulse-glow {
        0%, 100% { box-shadow: 0 0 5px rgba(239, 68, 68, 0.3); }
        50% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.6); }
    }

    @keyframes shimmer {
        0% { background-position: -200% 0; }
        100% { background-position: 200% 0; }
    }

    .animate-in {
        animation: fadeInUp 0.5s ease forwards;
    }

    .pulse-danger {
        animation: pulse-glow 2s ease-in-out infinite;
    }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }

    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }

    ::-webkit-scrollbar-thumb {
        background: var(--text-muted);
        border-radius: 3px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 20px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border-glass);
    }

    .section-header h2 {
        margin: 0 !important;
        padding: 0 !important;
    }

    .section-icon {
        width: 40px;
        height: 40px;
        border-radius: var(--radius-md);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.2rem;
    }

    /* ── Selectbox / Input Styling ── */
    .stSelectbox, .stTextInput, .stNumberInput {
        font-family: var(--font-main) !important;
    }

    /* ── Divider ── */
    .stDivider {
        border-color: var(--border-glass) !important;
    }

    /* ── Alert Info Box ── */
    .info-banner {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1), rgba(139, 92, 246, 0.1));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: var(--radius-md);
        padding: 16px 20px;
        color: var(--text-secondary);
        font-size: 0.9rem;
        line-height: 1.5;
    }

    .info-banner strong {
        color: var(--accent-blue);
    }

    /* ── Nav Item Card ── */
    .nav-item {
        padding: 12px 16px;
        border-radius: var(--radius-md);
        cursor: pointer;
        transition: all 0.2s ease;
        border: 1px solid transparent;
        margin-bottom: 4px;
    }

    .nav-item:hover {
        background: rgba(99, 102, 241, 0.1);
        border-color: rgba(99, 102, 241, 0.2);
    }

    .nav-item.active {
        background: rgba(99, 102, 241, 0.15);
        border-color: var(--accent-blue);
    }

    /* ── Hide Streamlit Branding ── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent !important; }

    /* ══════════════════════════════════════
       FIX: Streamlit Native Icon Visibility
       Dark theme makes built-in SVG icons
       invisible — force them to light colors.
       ══════════════════════════════════════ */

    /* ── Sidebar collapse / close button (✕ / ◀) ── */
    button[data-testid="stSidebarCollapseButton"],
    button[data-testid="baseButton-headerNoPadding"] {
        color: var(--text-primary) !important;
        opacity: 1 !important;
    }

    button[data-testid="stSidebarCollapseButton"] svg,
    button[data-testid="baseButton-headerNoPadding"] svg,
    section[data-testid="stSidebar"] button[kind="headerNoPadding"] svg {
        fill: var(--text-primary) !important;
        stroke: var(--text-primary) !important;
        color: var(--text-primary) !important;
        opacity: 1 !important;
    }

    /* ── Sidebar expand button (when sidebar is collapsed) ── */
    button[data-testid="stSidebarNavCollapseButton"] svg,
    button[data-testid="collapsedControl"] svg,
    [data-testid="collapsedControl"] svg {
        fill: var(--text-primary) !important;
        stroke: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }

    [data-testid="collapsedControl"] {
        color: var(--text-primary) !important;
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border-glass) !important;
    }

    /* ── File uploader — upload icon, text, and browse button ── */
    .stFileUploader svg,
    [data-testid="stFileUploader"] svg {
        fill: var(--accent-blue) !important;
        stroke: var(--accent-blue) !important;
        color: var(--accent-blue) !important;
        opacity: 1 !important;
    }

    .stFileUploader label,
    [data-testid="stFileUploader"] label {
        color: var(--text-primary) !important;
    }

    .stFileUploader small,
    [data-testid="stFileUploader"] small,
    .stFileUploader [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--text-secondary) !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: rgba(99, 102, 241, 0.05) !important;
        border-color: var(--border-glow) !important;
    }

    /* File uploader — delete/remove file button (✕) */
    .stFileUploader button[aria-label="Delete file"] svg,
    [data-testid="stFileUploader"] button svg,
    .stFileUploader [data-testid="baseButton-minimal"] svg {
        fill: var(--text-secondary) !important;
        stroke: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }

    .stFileUploader button[aria-label="Delete file"]:hover svg,
    .stFileUploader [data-testid="baseButton-minimal"]:hover svg {
        fill: var(--danger) !important;
        stroke: var(--danger) !important;
        color: var(--danger) !important;
    }

    /* ── Expander toggle arrows ── */
    .streamlit-expanderHeader svg,
    [data-testid="stExpander"] summary svg,
    [data-testid="stExpander"] [data-testid="stExpanderToggleIcon"] svg,
    details summary svg {
        fill: var(--text-secondary) !important;
        stroke: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
        opacity: 1 !important;
    }

    [data-testid="stExpander"] summary:hover svg,
    details summary:hover svg {
        fill: var(--accent-blue) !important;
        stroke: var(--accent-blue) !important;
        color: var(--accent-blue) !important;
    }

    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        color: var(--text-primary) !important;
    }

    /* ── Selectbox / dropdown arrows ── */
    .stSelectbox svg,
    [data-testid="stSelectbox"] svg,
    [data-baseweb="select"] svg {
        fill: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }

    [data-baseweb="select"] {
        color: var(--text-primary) !important;
    }

    /* ── Multiselect remove-tag (✕) icons ── */
    [data-baseweb="tag"] svg {
        fill: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }

    /* ── Radio buttons & checkboxes ── */
    .stRadio label span,
    .stCheckbox label span {
        color: var(--text-primary) !important;
    }

    .stRadio svg,
    .stCheckbox svg {
        fill: var(--accent-blue) !important;
        color: var(--accent-blue) !important;
    }

    /* ── Number input stepper arrows ── */
    .stNumberInput button svg,
    [data-testid="stNumberInput"] button svg {
        fill: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }

    /* ── Download button icon ── */
    .stDownloadButton svg,
    [data-testid="stDownloadButton"] svg {
        fill: white !important;
        color: white !important;
    }

    /* ── Toast / alert close buttons ── */
    [data-testid="stNotification"] button svg,
    .stAlert button svg {
        fill: var(--text-secondary) !important;
        color: var(--text-secondary) !important;
    }

    /* ── Modal / dialog close button ── */
    [data-testid="stModal"] button svg {
        fill: var(--text-primary) !important;
        color: var(--text-primary) !important;
    }

    /* ── Catch-all: any SVG inside a Streamlit button ── */
    .stApp button svg {
        fill: currentColor;
        color: inherit;
        opacity: 1 !important;
    }

    /* ── Ensure sidebar nav buttons always visible ── */
    section[data-testid="stSidebar"] button svg {
        fill: var(--text-primary) !important;
        stroke: var(--text-primary) !important;
        opacity: 1 !important;
    }
    </style>
    """


def metric_card_html(label, value, card_type="total", delta=None, icon=""):
    """Generate a premium metric card HTML."""
    delta_html = ""
    if delta is not None:
        delta_color = "var(--success)" if "+" not in str(delta) else "var(--danger)"
        delta_html = f'<div class="metric-delta" style="color: {delta_color}">{delta}</div>'

    return f"""
    <div class="metric-card {card_type} animate-in">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """


def decision_badge(decision: str) -> str:
    """Generate a styled decision badge."""
    badge_map = {
        "BLOCK": ("badge-block", "🔴 BLOCK"),
        "BLOCK_NOVEL": ("badge-novel", "🟠 NOVEL"),
        "REVIEW": ("badge-review", "🟡 REVIEW"),
        "ALLOW": ("badge-allow", "🟢 ALLOW"),
        "FRAUD": ("badge-block", "🔴 FRAUD"),
        "LEGIT": ("badge-allow", "🟢 LEGIT"),
    }
    cls, label = badge_map.get(decision, ("badge-allow", decision))
    return f'<span class="badge {cls}">{label}</span>'


def get_plotly_theme():
    """Get consistent Plotly chart theme."""
    return dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#94a3b8"),
        title_font=dict(family="Inter, sans-serif", color="#f1f5f9", size=16),
        colorway=[
            "#6366f1", "#8b5cf6", "#06b6d4", "#10b981",
            "#f59e0b", "#ef4444", "#ec4899", "#f97316"
        ],
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)",
            zerolinecolor="rgba(255,255,255,0.1)",
        ),
        margin=dict(l=40, r=20, t=50, b=40),
    )
