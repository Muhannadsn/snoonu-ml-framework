"""
Snoonu ML Framework - UI Design System
=======================================
Custom styling, components, and design utilities for the dashboard.
"""

import streamlit as st
import base64
from pathlib import Path


# =============================================================================
# BRAND COLORS
# =============================================================================

COLORS = {
    # Primary
    'primary': '#FF6B35',           # Snoonu Orange
    'primary_hover': '#FF8B5E',     # Lighter orange
    'primary_dark': '#E55A2B',      # Darker orange

    # Backgrounds
    'bg_dark': '#0E1117',           # Main background
    'bg_card': '#1A1F2E',           # Card background
    'bg_card_hover': '#242B3D',     # Card hover
    'bg_input': '#262D40',          # Input fields

    # Text
    'text_primary': '#FAFAFA',      # Main text
    'text_secondary': '#B0B8C8',    # Secondary text
    'text_muted': '#6B7280',        # Muted text

    # Accents
    'success': '#10B981',           # Green
    'warning': '#F59E0B',           # Yellow
    'error': '#EF4444',             # Red
    'info': '#3B82F6',              # Blue

    # Chart colors
    'chart_1': '#FF6B35',           # Orange
    'chart_2': '#3B82F6',           # Blue
    'chart_3': '#10B981',           # Green
    'chart_4': '#F59E0B',           # Yellow
    'chart_5': '#8B5CF6',           # Purple
    'chart_6': '#EC4899',           # Pink
    'chart_7': '#06B6D4',           # Cyan
    'chart_8': '#84CC16',           # Lime
}

# Chart color sequence for Plotly
CHART_COLORS = [
    COLORS['chart_1'], COLORS['chart_2'], COLORS['chart_3'],
    COLORS['chart_4'], COLORS['chart_5'], COLORS['chart_6'],
    COLORS['chart_7'], COLORS['chart_8']
]


# =============================================================================
# FONT LOADING
# =============================================================================

def load_fonts():
    """Load custom Altform fonts from local files."""
    font_dir = Path(__file__).parent / 'fonts'

    font_faces = ""

    font_files = {
        'Altform-Regular': 'Altform-Regular.otf',
        'Altform-Bold': 'Altform-Bold.otf',
        'Altform-Light': 'Altform-Light.otf',
        'Altform-Black': 'Altform-Black.otf',
    }

    for font_name, font_file in font_files.items():
        font_path = font_dir / font_file
        if font_path.exists():
            with open(font_path, 'rb') as f:
                font_data = base64.b64encode(f.read()).decode()

            weight = '400'
            if 'Bold' in font_name:
                weight = '700'
            elif 'Light' in font_name:
                weight = '300'
            elif 'Black' in font_name:
                weight = '900'

            font_faces += f"""
            @font-face {{
                font-family: 'Altform';
                src: url(data:font/otf;base64,{font_data}) format('opentype');
                font-weight: {weight};
                font-style: normal;
            }}
            """

    return font_faces


# =============================================================================
# MAIN CSS
# =============================================================================

def get_custom_css():
    """Get comprehensive custom CSS for the dashboard."""

    font_faces = load_fonts()

    return f"""
    <style>
    /* ========== FONT FACES ========== */
    {font_faces}

    /* ========== GLOBAL STYLES ========== */
    *:not([class*="icon"]):not([class*="material"]):not([data-testid="stIcon"]):not([translate="no"]) {{
        font-family: 'Altform', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }}

    /* Hide Streamlit branding but keep header for sidebar toggle */
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}

    /* Main container */
    .main .block-container {{
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }}

    /* ========== TYPOGRAPHY ========== */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Altform', sans-serif !important;
        font-weight: 700 !important;
        color: {COLORS['text_primary']} !important;
    }}

    h1 {{
        font-size: 2.5rem !important;
        font-weight: 900 !important;
        letter-spacing: -0.02em;
        margin-bottom: 1.5rem !important;
    }}

    h2 {{
        font-size: 1.75rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }}

    h3 {{
        font-size: 1.25rem !important;
        color: {COLORS['text_secondary']} !important;
    }}

    p, span, div {{
        color: {COLORS['text_primary']};
    }}

    /* ========== SIDEBAR ========== */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {COLORS['bg_card']} 0%, {COLORS['bg_dark']} 100%);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }}

    [data-testid="stSidebar"] .block-container {{
        padding-top: 2rem;
    }}

    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {{
        color: {COLORS['text_primary']} !important;
    }}

    /* Sidebar subheader styling */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2 {{
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: {COLORS['text_muted']} !important;
        margin-top: 1.5rem !important;
        margin-bottom: 0.75rem !important;
        font-weight: 600 !important;
    }}

    /* ========== CARDS ========== */
    .stCard {{
        background: {COLORS['bg_card']};
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }}

    .stCard:hover {{
        border-color: rgba(255, 107, 53, 0.3);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
    }}

    /* ========== METRICS ========== */
    [data-testid="stMetric"] {{
        background: {COLORS['bg_card']};
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }}

    [data-testid="stMetricLabel"] {{
        font-size: 0.875rem !important;
        font-weight: 500 !important;
        color: {COLORS['text_secondary']} !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }}

    [data-testid="stMetricValue"] {{
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: {COLORS['text_primary']} !important;
    }}

    [data-testid="stMetricDelta"] {{
        font-size: 0.875rem !important;
    }}

    /* ========== BUTTONS ========== */
    .stButton > button {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        font-size: 0.875rem;
        transition: all 0.2s ease;
        box-shadow: 0 2px 4px rgba(255, 107, 53, 0.3);
    }}

    .stButton > button:hover {{
        background: linear-gradient(135deg, {COLORS['primary_hover']} 0%, {COLORS['primary']} 100%);
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4);
        transform: translateY(-1px);
    }}

    .stButton > button:active {{
        transform: translateY(0);
    }}

    /* Secondary button style */
    .stButton > button[kind="secondary"] {{
        background: transparent;
        border: 1px solid {COLORS['primary']};
        color: {COLORS['primary']};
    }}

    /* ========== INPUTS ========== */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stTextArea > div > div > textarea {{
        background: {COLORS['bg_input']} !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: {COLORS['text_primary']} !important;
        padding: 0.75rem 1rem !important;
    }}

    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {{
        border-color: {COLORS['primary']} !important;
        box-shadow: 0 0 0 2px rgba(255, 107, 53, 0.2) !important;
    }}

    /* ========== SELECT BOXES ========== */
    .stSelectbox > div > div {{
        background: {COLORS['bg_input']} !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
    }}

    .stSelectbox > div > div:hover {{
        border-color: {COLORS['primary']} !important;
    }}

    /* ========== SLIDERS ========== */
    .stSlider > div > div > div > div {{
        background: {COLORS['primary']} !important;
    }}

    .stSlider > div > div > div > div > div {{
        background: {COLORS['primary']} !important;
    }}

    /* ========== TABS ========== */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0.5rem;
        background: {COLORS['bg_card']};
        border-radius: 12px;
        padding: 0.5rem;
    }}

    .stTabs [data-baseweb="tab"] {{
        background: transparent;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: {COLORS['text_secondary']};
        border: none;
    }}

    .stTabs [aria-selected="true"] {{
        background: {COLORS['primary']} !important;
        color: white !important;
    }}

    .stTabs [data-baseweb="tab"]:hover {{
        color: {COLORS['text_primary']};
        background: rgba(255, 255, 255, 0.05);
    }}

    /* ========== DATAFRAMES ========== */
    .stDataFrame {{
        border-radius: 12px;
        overflow: hidden;
    }}

    .stDataFrame [data-testid="stDataFrameContainer"] {{
        background: {COLORS['bg_card']};
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
    }}

    /* Table header */
    .stDataFrame thead th {{
        background: {COLORS['bg_dark']} !important;
        color: {COLORS['text_secondary']} !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        font-size: 0.75rem !important;
        letter-spacing: 0.05em;
    }}

    /* Table rows */
    .stDataFrame tbody td {{
        background: {COLORS['bg_card']} !important;
        color: {COLORS['text_primary']} !important;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important;
    }}

    .stDataFrame tbody tr:hover td {{
        background: {COLORS['bg_card_hover']} !important;
    }}

    /* ========== EXPANDERS ========== */
    .streamlit-expanderHeader {{
        background: {COLORS['bg_card']} !important;
        border-radius: 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        font-weight: 600 !important;
    }}

    .streamlit-expanderHeader:hover {{
        border-color: {COLORS['primary']} !important;
    }}

    .streamlit-expanderContent {{
        background: {COLORS['bg_card']} !important;
        border-radius: 0 0 8px 8px !important;
        border: 1px solid rgba(255, 255, 255, 0.05) !important;
        border-top: none !important;
    }}

    /* ========== ALERTS ========== */
    .stAlert {{
        border-radius: 8px;
        border: none;
    }}

    /* Success */
    [data-testid="stAlert"][data-baseweb="notification"] {{
        background: rgba(16, 185, 129, 0.1);
        border-left: 4px solid {COLORS['success']};
    }}

    /* ========== DIVIDERS ========== */
    hr {{
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
        margin: 2rem 0;
    }}

    /* ========== DOWNLOAD BUTTON ========== */
    .stDownloadButton > button {{
        background: transparent !important;
        border: 1px solid {COLORS['primary']} !important;
        color: {COLORS['primary']} !important;
    }}

    .stDownloadButton > button:hover {{
        background: rgba(255, 107, 53, 0.1) !important;
    }}

    /* ========== PROGRESS BAR ========== */
    .stProgress > div > div > div > div {{
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['chart_5']});
    }}

    /* ========== SPINNER ========== */
    .stSpinner > div {{
        border-top-color: {COLORS['primary']} !important;
    }}

    /* ========== CUSTOM CLASSES ========== */

    /* Gradient text */
    .gradient-text {{
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['chart_5']} 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }}

    /* Stat card */
    .stat-card {{
        background: linear-gradient(135deg, {COLORS['bg_card']} 0%, {COLORS['bg_dark']} 100%);
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.05);
        position: relative;
        overflow: hidden;
    }}

    .stat-card::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, {COLORS['primary']}, {COLORS['chart_5']});
    }}

    .stat-card-value {{
        font-size: 2.5rem;
        font-weight: 900;
        color: {COLORS['text_primary']};
        line-height: 1;
        margin-bottom: 0.5rem;
    }}

    .stat-card-label {{
        font-size: 0.875rem;
        color: {COLORS['text_secondary']};
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }}

    .stat-card-delta {{
        font-size: 0.875rem;
        margin-top: 0.5rem;
    }}

    .stat-card-delta.positive {{
        color: {COLORS['success']};
    }}

    .stat-card-delta.negative {{
        color: {COLORS['error']};
    }}

    /* Section header */
    .section-header {{
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin-bottom: 1.5rem;
    }}

    .section-header h2 {{
        margin: 0 !important;
    }}

    .section-header .icon {{
        width: 40px;
        height: 40px;
        background: linear-gradient(135deg, {COLORS['primary']} 0%, {COLORS['primary_dark']} 100%);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.25rem;
    }}

    /* Page header */
    .page-header {{
        margin-bottom: 2rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}

    .page-header h1 {{
        margin-bottom: 0.5rem !important;
    }}

    .page-header p {{
        color: {COLORS['text_secondary']};
        font-size: 1.1rem;
        margin: 0;
    }}

    /* Navigation pill */
    .nav-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: {COLORS['bg_card']};
        border-radius: 20px;
        font-size: 0.875rem;
        color: {COLORS['text_secondary']};
        text-decoration: none;
        transition: all 0.2s ease;
    }}

    .nav-pill:hover {{
        background: {COLORS['bg_card_hover']};
        color: {COLORS['text_primary']};
    }}

    .nav-pill.active {{
        background: {COLORS['primary']};
        color: white;
    }}

    /* Logo area */
    .logo-container {{
        padding: 1rem 0 1.5rem 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
    }}

    .logo-text {{
        font-size: 1.5rem;
        font-weight: 900;
        color: {COLORS['text_primary']};
        letter-spacing: -0.02em;
    }}

    .logo-text span {{
        color: {COLORS['primary']};
    }}

    .logo-subtitle {{
        font-size: 0.75rem;
        color: {COLORS['text_muted']};
        text-transform: uppercase;
        letter-spacing: 0.15em;
        margin-top: 0.25rem;
    }}

    </style>
    """


# =============================================================================
# COMPONENT FUNCTIONS
# =============================================================================

def apply_theme():
    """Apply the complete custom theme to the dashboard."""
    st.markdown(get_custom_css(), unsafe_allow_html=True)


def render_logo():
    """Render the Snoonu ML logo in sidebar."""
    st.markdown("""
        <div class="logo-container">
            <div class="logo-text">Snoonu<span>ML</span></div>
            <div class="logo-subtitle">Analytics Platform</div>
        </div>
    """, unsafe_allow_html=True)


def render_loading_screen():
    """Render a branded loading screen with animation."""
    st.markdown("""
        <style>
        .loading-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 60vh;
            text-align: center;
        }
        .loading-logo {
            font-family: 'Altform', sans-serif;
            font-size: 3rem;
            font-weight: 900;
            color: #FAFAFA;
            margin-bottom: 2rem;
        }
        .loading-logo span {
            color: #FF6B35;
        }
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 4px solid #1A1F2E;
            border-top: 4px solid #FF6B35;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin-bottom: 1.5rem;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .loading-text {
            color: #B0B8C8;
            font-size: 1rem;
            animation: pulse 1.5s ease-in-out infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 0.6; }
            50% { opacity: 1; }
        }
        .loading-dots {
            display: inline-flex;
            gap: 4px;
            margin-left: 4px;
        }
        .loading-dots span {
            width: 6px;
            height: 6px;
            background: #FF6B35;
            border-radius: 50%;
            animation: bounce 1.4s ease-in-out infinite;
        }
        .loading-dots span:nth-child(1) { animation-delay: 0s; }
        .loading-dots span:nth-child(2) { animation-delay: 0.2s; }
        .loading-dots span:nth-child(3) { animation-delay: 0.4s; }
        @keyframes bounce {
            0%, 80%, 100% { transform: translateY(0); }
            40% { transform: translateY(-8px); }
        }
        </style>
        <div class="loading-container">
            <div class="loading-logo">Snoonu<span>ML</span></div>
            <div class="loading-spinner"></div>
            <div class="loading-text">
                Loading your analytics
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_page_header(title: str, description: str = None):
    """Render a styled page header."""
    st.markdown(f"# {title}")
    if description:
        st.caption(description)
    st.markdown("---")


def render_section_header(title: str, icon: str = None):
    """Render a styled section header with optional icon."""
    if icon:
        st.markdown(f"### {icon} {title}")
    else:
        st.markdown(f"### {title}")


def render_stat_card(value: str, label: str, delta: str = None, delta_positive: bool = True):
    """Render a custom stat card."""
    delta_html = ''
    if delta:
        delta_class = 'positive' if delta_positive else 'negative'
        delta_symbol = '+' if delta_positive else ''
        delta_html = f'<div class="stat-card-delta {delta_class}">{delta_symbol}{delta}</div>'

    st.markdown(f"""
        <div class="stat-card">
            <div class="stat-card-value">{value}</div>
            <div class="stat-card-label">{label}</div>
            {delta_html}
        </div>
    """, unsafe_allow_html=True)


def render_metric_row(metrics: list):
    """
    Render a row of metric cards.

    Args:
        metrics: List of dicts with keys: value, label, delta (optional), delta_positive (optional)
    """
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            render_stat_card(
                value=metric['value'],
                label=metric['label'],
                delta=metric.get('delta'),
                delta_positive=metric.get('delta_positive', True)
            )


def create_card(content_func):
    """
    Decorator to wrap content in a card.

    Usage:
        with create_card():
            st.write("Content here")
    """
    st.markdown('<div class="stCard">', unsafe_allow_html=True)
    content_func()
    st.markdown('</div>', unsafe_allow_html=True)


# =============================================================================
# PLOTLY THEME
# =============================================================================

def get_plotly_theme():
    """Get consistent Plotly theme configuration."""
    return {
        'template': 'plotly_dark',
        'color_discrete_sequence': CHART_COLORS,
        'layout': {
            'paper_bgcolor': 'rgba(0,0,0,0)',
            'plot_bgcolor': 'rgba(0,0,0,0)',
            'font': {
                'family': 'Altform, sans-serif',
                'color': COLORS['text_primary'],
                'size': 12
            },
            'title': {
                'font': {
                    'size': 18,
                    'color': COLORS['text_primary']
                },
                'x': 0,
                'xanchor': 'left'
            },
            'legend': {
                'bgcolor': 'rgba(0,0,0,0)',
                'font': {'color': COLORS['text_secondary']}
            },
            'xaxis': {
                'gridcolor': 'rgba(255,255,255,0.05)',
                'linecolor': 'rgba(255,255,255,0.1)',
                'tickfont': {'color': COLORS['text_secondary']}
            },
            'yaxis': {
                'gridcolor': 'rgba(255,255,255,0.05)',
                'linecolor': 'rgba(255,255,255,0.1)',
                'tickfont': {'color': COLORS['text_secondary']}
            },
            'margin': {'l': 40, 'r': 20, 't': 60, 'b': 40}
        }
    }


def apply_plotly_theme(fig):
    """Apply consistent theme to a Plotly figure."""
    theme = get_plotly_theme()

    fig.update_layout(
        paper_bgcolor=theme['layout']['paper_bgcolor'],
        plot_bgcolor=theme['layout']['plot_bgcolor'],
        font=theme['layout']['font'],
        legend=theme['layout']['legend'],
        margin=theme['layout']['margin']
    )

    fig.update_xaxes(
        gridcolor=theme['layout']['xaxis']['gridcolor'],
        linecolor=theme['layout']['xaxis']['linecolor'],
        tickfont=theme['layout']['xaxis']['tickfont']
    )

    fig.update_yaxes(
        gridcolor=theme['layout']['yaxis']['gridcolor'],
        linecolor=theme['layout']['yaxis']['linecolor'],
        tickfont=theme['layout']['yaxis']['tickfont']
    )

    return fig


def styled_bar_chart(df, x, y, title=None, orientation='v', **kwargs):
    """Create a styled bar chart."""
    import plotly.express as px

    fig = px.bar(
        df, x=x, y=y,
        title=title,
        orientation=orientation,
        color_discrete_sequence=CHART_COLORS,
        **kwargs
    )

    return apply_plotly_theme(fig)


def styled_line_chart(df, x, y, title=None, **kwargs):
    """Create a styled line chart."""
    import plotly.express as px

    fig = px.line(
        df, x=x, y=y,
        title=title,
        color_discrete_sequence=CHART_COLORS,
        **kwargs
    )

    fig.update_traces(line=dict(width=2.5))

    return apply_plotly_theme(fig)


def styled_pie_chart(df, values, names, title=None, **kwargs):
    """Create a styled pie/donut chart."""
    import plotly.express as px

    fig = px.pie(
        df, values=values, names=names,
        title=title,
        color_discrete_sequence=CHART_COLORS,
        hole=0.4,  # Donut style
        **kwargs
    )

    fig.update_traces(
        textposition='outside',
        textfont=dict(color=COLORS['text_primary'])
    )

    return apply_plotly_theme(fig)


def styled_scatter_chart(df, x, y, title=None, **kwargs):
    """Create a styled scatter chart."""
    import plotly.express as px

    fig = px.scatter(
        df, x=x, y=y,
        title=title,
        color_discrete_sequence=CHART_COLORS,
        **kwargs
    )

    return apply_plotly_theme(fig)


# =============================================================================
# NAVIGATION
# =============================================================================

def render_nav_menu(options: list, icons: list = None, default: int = 0):
    """
    Render a styled navigation menu.

    Args:
        options: List of menu option strings
        icons: Optional list of icons for each option
        default: Default selected index

    Returns:
        Selected option string
    """
    # Use streamlit-option-menu if available, otherwise fallback to selectbox
    try:
        from streamlit_option_menu import option_menu

        selected = option_menu(
            menu_title=None,
            options=options,
            icons=icons or ['circle'] * len(options),
            default_index=default,
            orientation="vertical",
            styles={
                "container": {
                    "padding": "0 !important",
                    "background-color": "transparent"
                },
                "icon": {
                    "color": COLORS['primary'],
                    "font-size": "16px"
                },
                "nav-link": {
                    "font-size": "14px",
                    "text-align": "left",
                    "margin": "2px 0",
                    "padding": "10px 15px",
                    "border-radius": "8px",
                    "color": COLORS['text_secondary'],
                    "background-color": "transparent",
                    "--hover-color": COLORS['bg_card_hover']
                },
                "nav-link-selected": {
                    "background-color": COLORS['primary'],
                    "color": "white",
                    "font-weight": "600"
                }
            }
        )
        return selected

    except ImportError:
        # Fallback to styled selectbox
        return st.selectbox("Navigation", options, index=default, label_visibility="collapsed")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_number(value, prefix='', suffix='', decimals=0):
    """Format a number with optional prefix/suffix."""
    if value >= 1_000_000:
        formatted = f"{value/1_000_000:.{decimals}f}M"
    elif value >= 1_000:
        formatted = f"{value/1_000:.{decimals}f}K"
    else:
        formatted = f"{value:,.{decimals}f}"

    return f"{prefix}{formatted}{suffix}"


def format_currency(value, currency='QAR'):
    """Format a value as currency."""
    return f"{currency} {value:,.0f}"


def format_percentage(value, decimals=1):
    """Format a value as percentage."""
    return f"{value:.{decimals}f}%"
