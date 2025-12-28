"""
SnoonuML UI Framework
=====================
Airbnb-inspired card gallery layout with top bar navigation.
"""

import streamlit as st
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum


class Category(Enum):
    CORE = "Core Analytics"
    ML = "Machine Learning"
    BUSINESS = "Business Intel"
    ADVANCED = "Advanced Intel"


# Category colors and gradients
CATEGORY_STYLES = {
    Category.CORE: {
        "gradient": "linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%)",
        "color": "#3B82F6",
        "icon": "📊"
    },
    Category.ML: {
        "gradient": "linear-gradient(135deg, #8B5CF6 0%, #6D28D9 100%)",
        "color": "#8B5CF6",
        "icon": "🤖"
    },
    Category.BUSINESS: {
        "gradient": "linear-gradient(135deg, #10B981 0%, #059669 100%)",
        "color": "#10B981",
        "icon": "💼"
    },
    Category.ADVANCED: {
        "gradient": "linear-gradient(135deg, #F59E0B 0%, #D97706 100%)",
        "color": "#F59E0B",
        "icon": "🔬"
    }
}


@dataclass
class ModuleCard:
    """Represents a module card in the gallery."""
    id: str
    title: str
    subtitle: str
    category: Category
    stats: Dict[str, str] = None
    badge: str = None  # e.g., "Popular", "New", "Beta"


# Define all modules with their categories
MODULES = [
    # Core Analytics
    ModuleCard("overview", "Overview", "Key metrics at a glance", Category.CORE),
    ModuleCard("session_analytics", "Session Analytics", "User session patterns", Category.CORE),
    ModuleCard("funnel_analysis", "Funnel Analysis", "Conversion funnel deep-dive", Category.CORE),
    ModuleCard("customer_journey", "Customer Journey", "Path visualization & friction", Category.CORE, badge="New"),
    ModuleCard("cohort_analysis", "Cohort Analysis", "Retention by cohort", Category.CORE),
    ModuleCard("anomaly_detection", "Anomaly Detection", "Spot unusual patterns", Category.CORE, badge="Alert"),

    # Machine Learning
    ModuleCard("ml_predictions", "ML Predictions", "Churn, Conversion, LTV models", Category.ML, badge="Popular"),
    ModuleCard("recommendations", "Recommendations", "Personalized suggestions", Category.ML),
    ModuleCard("customer_scoring", "Customer Scoring", "RFM & value scoring", Category.ML),
    ModuleCard("survival_analysis", "Survival Analysis", "Time-to-event modeling", Category.ML, badge="New"),

    # Business Intel
    ModuleCard("merchant_intelligence", "Merchant Intelligence", "Restaurant performance", Category.BUSINESS),
    ModuleCard("promo_analytics", "Promo Analytics", "Campaign effectiveness", Category.BUSINESS),
    ModuleCard("search_analytics", "Search Analytics", "What users look for", Category.BUSINESS),
    ModuleCard("delivery_analytics", "Delivery Analytics", "Fulfillment metrics", Category.BUSINESS),
    ModuleCard("order_analysis", "Order Analysis", "Order patterns & trends", Category.BUSINESS),

    # Advanced Intel
    ModuleCard("attribution_modeling", "Attribution Modeling", "Channel contribution", Category.ADVANCED),
    ModuleCard("reactivation_targeting", "Reactivation Targeting", "Win-back opportunities", Category.ADVANCED),
    ModuleCard("product_affinity", "Product Affinity", "Cross-sell insights", Category.ADVANCED),
    ModuleCard("trending", "Trending", "What's hot right now", Category.ADVANCED),
    ModuleCard("platform_analysis", "Platform Analysis", "iOS vs Android", Category.ADVANCED),
    ModuleCard("hourly_trends", "Hourly Trends", "Time-based patterns", Category.ADVANCED),
]


def get_framework_css() -> str:
    """Get the CSS for the new UI framework."""
    return """
    <style>
    /* ============================================
       AIRBNB-INSPIRED UI FRAMEWORK
       ============================================ */

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header[data-testid="stHeader"] {display: none;}
    .stDeployButton {display: none;}

    /* Reset padding for full-width top bar */
    .main .block-container {
        padding-top: 0 !important;
        max-width: 100% !important;
    }

    /* ============================================
       TOP BAR
       ============================================ */
    .top-bar {
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        height: 72px;
        background: #0E1117;
        border-bottom: 1px solid rgba(255,255,255,0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0 24px;
        z-index: 1000;
        backdrop-filter: blur(10px);
    }

    .top-bar-left {
        display: flex;
        align-items: center;
        gap: 24px;
    }

    .top-bar-logo {
        font-family: 'Altform', sans-serif;
        font-size: 1.5rem;
        font-weight: 900;
        color: #FAFAFA;
        cursor: pointer;
        transition: opacity 0.2s;
    }

    .top-bar-logo:hover {
        opacity: 0.8;
    }

    .top-bar-logo span {
        color: #FF6B35;
    }

    .breadcrumb {
        display: flex;
        align-items: center;
        gap: 8px;
        color: #888;
        font-size: 0.9rem;
    }

    .breadcrumb a {
        color: #888;
        text-decoration: none;
        transition: color 0.2s;
    }

    .breadcrumb a:hover {
        color: #FF6B35;
    }

    .breadcrumb-current {
        color: #FAFAFA;
    }

    /* ============================================
       SEARCH PILL
       ============================================ */
    .top-bar-center {
        flex: 1;
        display: flex;
        justify-content: center;
        max-width: 600px;
        margin: 0 auto;
    }

    .search-pill {
        display: flex;
        align-items: center;
        background: #1A1F2E;
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 40px;
        padding: 8px 8px 8px 20px;
        width: 100%;
        max-width: 500px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .search-pill:hover, .search-pill:focus-within {
        border-color: rgba(255,107,53,0.5);
        box-shadow: 0 4px 16px rgba(0,0,0,0.3);
    }

    .search-pill input {
        flex: 1;
        background: transparent;
        border: none;
        color: #FAFAFA;
        font-size: 0.95rem;
        outline: none;
    }

    .search-pill input::placeholder {
        color: #666;
    }

    .search-pill-btn {
        background: #FF6B35;
        border: none;
        border-radius: 50%;
        width: 36px;
        height: 36px;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        transition: background 0.2s;
    }

    .search-pill-btn:hover {
        background: #FF8B5E;
    }

    /* ============================================
       TOP BAR RIGHT
       ============================================ */
    .top-bar-right {
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .user-menu {
        display: flex;
        align-items: center;
        gap: 8px;
        background: #1A1F2E;
        padding: 6px 12px 6px 6px;
        border-radius: 24px;
        border: 1px solid rgba(255,255,255,0.1);
        cursor: pointer;
        transition: border-color 0.2s;
    }

    .user-menu:hover {
        border-color: rgba(255,255,255,0.3);
    }

    .user-avatar {
        width: 32px;
        height: 32px;
        background: linear-gradient(135deg, #FF6B35, #FF8B5E);
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: 700;
        font-size: 0.85rem;
        color: white;
    }

    .user-name {
        color: #FAFAFA;
        font-size: 0.9rem;
    }

    /* ============================================
       MAIN CONTENT AREA
       ============================================ */
    .main-content {
        margin-top: 72px;
        padding: 32px 48px;
        min-height: calc(100vh - 72px);
    }

    /* ============================================
       DISCOVERY ROWS (CATEGORY SECTIONS)
       ============================================ */
    .discovery-section {
        margin-bottom: 48px;
    }

    .discovery-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 20px;
    }

    .discovery-title {
        font-family: 'Altform', sans-serif;
        font-size: 1.4rem;
        font-weight: 700;
        color: #FAFAFA;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .discovery-title-icon {
        font-size: 1.2rem;
    }

    .discovery-show-all {
        color: #888;
        font-size: 0.9rem;
        cursor: pointer;
        transition: color 0.2s;
    }

    .discovery-show-all:hover {
        color: #FF6B35;
    }

    /* ============================================
       HORIZONTAL SCROLL ROW
       ============================================ */
    .card-row {
        display: flex;
        gap: 20px;
        overflow-x: auto;
        scroll-behavior: smooth;
        padding: 4px 0 20px 0;
        margin: 0 -4px;
        scrollbar-width: none;
        -ms-overflow-style: none;
    }

    .card-row::-webkit-scrollbar {
        display: none;
    }

    /* ============================================
       MODULE CARD
       ============================================ */
    .module-card {
        flex: 0 0 280px;
        background: #1A1F2E;
        border-radius: 16px;
        overflow: hidden;
        cursor: pointer;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .module-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.4);
        border-color: rgba(255,107,53,0.3);
    }

    .module-card:active {
        transform: translateY(-2px);
    }

    .card-thumbnail {
        height: 140px;
        position: relative;
        overflow: hidden;
    }

    .card-thumbnail-gradient {
        width: 100%;
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }

    .card-thumbnail-icon {
        font-size: 3rem;
        opacity: 0.3;
    }

    .card-badge {
        position: absolute;
        top: 12px;
        left: 12px;
        background: rgba(0,0,0,0.6);
        backdrop-filter: blur(4px);
        color: #FAFAFA;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .card-badge-popular {
        background: rgba(255,107,53,0.9);
    }

    .card-badge-new {
        background: rgba(16,185,129,0.9);
    }

    .card-badge-alert {
        background: rgba(239,68,68,0.9);
    }

    .card-content {
        padding: 16px;
    }

    .card-title {
        font-family: 'Altform', sans-serif;
        font-size: 1rem;
        font-weight: 600;
        color: #FAFAFA;
        margin-bottom: 4px;
    }

    .card-subtitle {
        font-size: 0.85rem;
        color: #888;
        margin-bottom: 12px;
    }

    .card-stats {
        display: flex;
        gap: 16px;
    }

    .card-stat {
        display: flex;
        flex-direction: column;
    }

    .card-stat-value {
        font-size: 0.9rem;
        font-weight: 600;
        color: #FAFAFA;
    }

    .card-stat-label {
        font-size: 0.75rem;
        color: #666;
    }

    /* ============================================
       SKELETON LOADERS
       ============================================ */
    @keyframes skeleton-pulse {
        0%, 100% { opacity: 0.4; }
        50% { opacity: 0.8; }
    }

    .skeleton {
        background: linear-gradient(90deg, #1A1F2E 25%, #242B3D 50%, #1A1F2E 75%);
        background-size: 200% 100%;
        animation: skeleton-pulse 1.5s ease-in-out infinite;
        border-radius: 8px;
    }

    .skeleton-card {
        flex: 0 0 280px;
        height: 240px;
        border-radius: 16px;
    }

    .skeleton-text {
        height: 16px;
        margin-bottom: 8px;
    }

    .skeleton-text-sm {
        height: 12px;
        width: 60%;
    }

    /* ============================================
       BACK BUTTON
       ============================================ */
    .back-button {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        color: #888;
        font-size: 0.9rem;
        cursor: pointer;
        padding: 8px 16px;
        border-radius: 8px;
        transition: all 0.2s;
        margin-bottom: 24px;
    }

    .back-button:hover {
        color: #FF6B35;
        background: rgba(255,107,53,0.1);
    }

    /* ============================================
       MODULE PAGE HEADER
       ============================================ */
    .module-header {
        margin-bottom: 32px;
    }

    .module-header-title {
        font-family: 'Altform', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: #FAFAFA;
        margin-bottom: 8px;
    }

    .module-header-subtitle {
        color: #888;
        font-size: 1rem;
    }

    /* ============================================
       RESPONSIVE
       ============================================ */
    @media (max-width: 768px) {
        .top-bar {
            padding: 0 16px;
        }

        .top-bar-center {
            display: none;
        }

        .main-content {
            padding: 24px 16px;
        }

        .module-card {
            flex: 0 0 260px;
        }

        .discovery-title {
            font-size: 1.2rem;
        }
    }

    /* ============================================
       SEARCH RESULTS HIGHLIGHT
       ============================================ */
    .module-card.search-match {
        border-color: rgba(255,107,53,0.5);
    }

    .module-card.search-hidden {
        display: none;
    }

    </style>
    """


def render_top_bar(username: str, current_page: str = None, on_home: Callable = None):
    """Render the top navigation bar."""

    # Build breadcrumb
    if current_page:
        breadcrumb_html = f"""
            <div class="breadcrumb">
                <a href="#" onclick="window.location.reload()">Home</a>
                <span>›</span>
                <span class="breadcrumb-current">{current_page}</span>
            </div>
        """
    else:
        breadcrumb_html = ""

    # Get user initials
    initials = username[0].upper() if username else "U"

    top_bar_html = f"""
    <div class="top-bar">
        <div class="top-bar-left">
            <div class="top-bar-logo" onclick="window.location.reload()">
                Snoonu<span>ML</span>
            </div>
            {breadcrumb_html}
        </div>

        <div class="top-bar-center">
            <div class="search-pill">
                <input type="text" id="global-search" placeholder="Search insights..." />
                <button class="search-pill-btn">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                        <circle cx="11" cy="11" r="8"/>
                        <path d="m21 21-4.35-4.35"/>
                    </svg>
                </button>
            </div>
        </div>

        <div class="top-bar-right">
            <div class="user-menu">
                <div class="user-avatar">{initials}</div>
                <span class="user-name">{username}</span>
            </div>
        </div>
    </div>
    """

    st.markdown(top_bar_html, unsafe_allow_html=True)


def render_module_card(module: ModuleCard, stats: Dict[str, str] = None) -> str:
    """Generate HTML for a module card."""

    style = CATEGORY_STYLES[module.category]

    # Badge HTML
    badge_html = ""
    if module.badge:
        badge_class = f"card-badge-{module.badge.lower()}"
        badge_html = f'<div class="card-badge {badge_class}">{module.badge}</div>'

    # Stats HTML
    stats_html = ""
    if stats:
        stats_items = "".join([
            f'''<div class="card-stat">
                <span class="card-stat-value">{value}</span>
                <span class="card-stat-label">{label}</span>
            </div>'''
            for label, value in stats.items()
        ])
        stats_html = f'<div class="card-stats">{stats_items}</div>'

    return f"""
    <div class="module-card" data-module="{module.id}" onclick="selectModule('{module.id}')">
        <div class="card-thumbnail">
            <div class="card-thumbnail-gradient" style="background: {style['gradient']};">
                <span class="card-thumbnail-icon">{style['icon']}</span>
            </div>
            {badge_html}
        </div>
        <div class="card-content">
            <div class="card-title">{module.title}</div>
            <div class="card-subtitle">{module.subtitle}</div>
            {stats_html}
        </div>
    </div>
    """


def render_skeleton_cards(count: int = 4) -> str:
    """Render skeleton loader cards."""
    cards = "".join(['<div class="skeleton skeleton-card"></div>' for _ in range(count)])
    return f'<div class="card-row">{cards}</div>'


def render_discovery_row(category: Category, modules: List[ModuleCard],
                         stats_func: Callable = None) -> str:
    """Render a discovery row for a category."""

    style = CATEGORY_STYLES[category]

    # Generate cards
    cards_html = ""
    for module in modules:
        # Get stats if function provided
        stats = stats_func(module.id) if stats_func else None
        cards_html += render_module_card(module, stats)

    return f"""
    <div class="discovery-section" data-category="{category.value}">
        <div class="discovery-header">
            <div class="discovery-title">
                <span class="discovery-title-icon">{style['icon']}</span>
                {category.value}
            </div>
            <span class="discovery-show-all">Show all →</span>
        </div>
        <div class="card-row">
            {cards_html}
        </div>
    </div>
    """


def render_gallery(stats_func: Callable = None) -> str:
    """Render the full gallery with all categories."""

    # Group modules by category
    categories = {}
    for module in MODULES:
        if module.category not in categories:
            categories[module.category] = []
        categories[module.category].append(module)

    # Render each category row
    gallery_html = ""
    for category in [Category.CORE, Category.ML, Category.BUSINESS, Category.ADVANCED]:
        if category in categories:
            gallery_html += render_discovery_row(category, categories[category], stats_func)

    return f"""
    <div class="main-content">
        {gallery_html}
    </div>
    """


def render_back_button() -> str:
    """Render the back to gallery button."""
    return """
    <div class="back-button" onclick="window.location.reload()">
        ← Back to Insights
    </div>
    """


def render_module_header(title: str, subtitle: str) -> str:
    """Render a module page header."""
    return f"""
    <div class="module-header">
        <div class="module-header-title">{title}</div>
        <div class="module-header-subtitle">{subtitle}</div>
    </div>
    """


def get_search_js() -> str:
    """Get JavaScript for search functionality."""
    return """
    <script>
    // Module selection
    function selectModule(moduleId) {
        // Update URL or trigger Streamlit
        const params = new URLSearchParams(window.location.search);
        params.set('module', moduleId);
        window.location.search = params.toString();
    }

    // Search functionality
    document.addEventListener('DOMContentLoaded', function() {
        const searchInput = document.getElementById('global-search');
        if (searchInput) {
            searchInput.addEventListener('input', function(e) {
                const query = e.target.value.toLowerCase();
                const cards = document.querySelectorAll('.module-card');

                cards.forEach(card => {
                    const title = card.querySelector('.card-title').textContent.toLowerCase();
                    const subtitle = card.querySelector('.card-subtitle').textContent.toLowerCase();

                    if (title.includes(query) || subtitle.includes(query) || query === '') {
                        card.classList.remove('search-hidden');
                        card.classList.add('search-match');
                    } else {
                        card.classList.add('search-hidden');
                        card.classList.remove('search-match');
                    }
                });
            });

            // Enter to jump to first result
            searchInput.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    const visibleCard = document.querySelector('.module-card:not(.search-hidden)');
                    if (visibleCard) {
                        visibleCard.click();
                    }
                }
            });
        }
    });
    </script>
    """


def init_framework():
    """Initialize the UI framework - call this at the start of app.py"""
    st.markdown(get_framework_css(), unsafe_allow_html=True)
    st.markdown(get_search_js(), unsafe_allow_html=True)
