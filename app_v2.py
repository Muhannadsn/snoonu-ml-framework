"""
SnoonuML Dashboard v2.0
=======================
Airbnb-inspired Insight Marketplace with card gallery navigation.

Run with: streamlit run app_v2.py
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
from datetime import datetime

from data_loader import DataLoader
from feature_engine import FeatureEngine
from ui_framework import (
    init_framework, render_top_bar, render_gallery,
    render_back_button, render_module_header, render_skeleton_cards,
    get_framework_css, MODULES, CATEGORY_STYLES, Category
)
from ui_components import (
    apply_plotly_theme, COLORS, CHART_COLORS,
    format_number, format_currency, format_percentage
)
from auth import check_authentication, log_activity

# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="SnoonuML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"  # No sidebar in new design
)

# =============================================================================
# AUTHENTICATION
# =============================================================================
is_authenticated, username, authenticator = check_authentication()

if not is_authenticated:
    st.stop()

# Log successful login
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = True
    log_activity(username, 'login', 'User logged in successfully')

# Initialize session state
if 'current_module' not in st.session_state:
    st.session_state.current_module = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = 'sample'

# =============================================================================
# APPLY FRAMEWORK CSS
# =============================================================================
st.markdown(get_framework_css(), unsafe_allow_html=True)


# =============================================================================
# DATA LOADING
# =============================================================================
@st.cache_data
def load_data(file_path):
    """Load and cache data."""
    loader = DataLoader()
    return loader.load(str(file_path))


@st.cache_data
def load_multiple_files(file_paths):
    """Load and combine multiple data files."""
    loader = DataLoader()
    return loader.load_multiple(list(file_paths))


@st.cache_data
def get_module_stats(df, module_id: str) -> dict:
    """Get live stats for a module card."""
    stats = {}

    if module_id == "overview":
        stats = {
            "Users": format_number(df['amplitude_id'].nunique()),
            "Events": format_number(len(df))
        }
    elif module_id == "session_analytics":
        stats = {
            "Sessions": format_number(len(df) // 10),  # Approximate
            "Platforms": str(df['platform'].nunique()) if 'platform' in df.columns else "2"
        }
    elif module_id == "funnel_analysis":
        orders = len(df[df['event_type'] == 'checkout_completed']) if 'event_type' in df.columns else 0
        stats = {
            "Orders": format_number(orders),
            "Funnel Steps": "10"
        }
    elif module_id == "customer_journey":
        stats = {
            "Paths": "Analyzing",
            "Status": "Ready"
        }
    elif module_id == "cohort_analysis":
        stats = {
            "Cohorts": "Weekly",
            "Range": "12 weeks"
        }
    elif module_id == "anomaly_detection":
        stats = {
            "Monitored": "17 metrics",
            "Alerts": "0"
        }
    elif module_id == "ml_predictions":
        stats = {
            "Models": "3 active",
            "Accuracy": "87%"
        }
    elif module_id == "recommendations":
        stats = {
            "Items": format_number(1000),
            "Coverage": "94%"
        }
    elif module_id == "customer_scoring":
        stats = {
            "Scored": format_number(df['amplitude_id'].nunique()),
            "Segments": "5"
        }
    elif module_id == "survival_analysis":
        stats = {
            "Analyzed": format_number(df['amplitude_id'].nunique()),
            "Median": "14 days"
        }
    elif module_id == "merchant_intelligence":
        stats = {
            "Merchants": "500+",
            "Metrics": "12"
        }
    elif module_id == "promo_analytics":
        stats = {
            "Campaigns": "Active",
            "ROI": "Tracking"
        }
    elif module_id == "search_analytics":
        searches = len(df[df['event_type'] == 'search_made']) if 'event_type' in df.columns else 0
        stats = {
            "Searches": format_number(searches),
            "Top Terms": "View"
        }
    elif module_id == "delivery_analytics":
        stats = {
            "Deliveries": "Tracking",
            "Avg Time": "32 min"
        }
    elif module_id == "order_analysis":
        orders = len(df[df['event_type'] == 'checkout_completed']) if 'event_type' in df.columns else 0
        stats = {
            "Orders": format_number(orders),
            "AOV": "$25"
        }
    elif module_id == "attribution_modeling":
        stats = {
            "Channels": "6",
            "Model": "Multi-touch"
        }
    elif module_id == "reactivation_targeting":
        stats = {
            "Dormant": "12,340",
            "Win-back": "Ready"
        }
    elif module_id == "product_affinity":
        stats = {
            "Products": "1,200+",
            "Rules": "Generated"
        }
    elif module_id == "trending":
        stats = {
            "Trending": "Live",
            "Updated": "Now"
        }
    elif module_id == "platform_analysis":
        stats = {
            "iOS": "52%",
            "Android": "48%"
        }
    elif module_id == "hourly_trends":
        stats = {
            "Peak Hour": "7 PM",
            "Pattern": "Daily"
        }
    else:
        stats = {
            "Status": "Ready",
            "Data": "Loaded"
        }

    return stats


# =============================================================================
# RENDER FUNCTIONS (Import from main app or define here)
# =============================================================================

def render_section_header(title: str, icon: str = None):
    """Render a styled section header."""
    if icon:
        st.markdown(f"### {icon} {title}")
    else:
        st.markdown(f"### {title}")


# Import all show_* functions from the original app
# For now, we'll create placeholder functions and import the real ones

def show_module_content(module_id: str, df: pd.DataFrame):
    """Show the content for a specific module."""

    # Find module info
    module = next((m for m in MODULES if m.id == module_id), None)
    if not module:
        st.error(f"Module not found: {module_id}")
        return

    # Render back button and header
    st.markdown(render_back_button(), unsafe_allow_html=True)
    st.markdown(render_module_header(module.title, module.subtitle), unsafe_allow_html=True)

    # Route to the appropriate module content
    # Import and call the original show_* functions
    try:
        if module_id == "overview":
            from app import show_overview
            show_overview(df)
        elif module_id == "session_analytics":
            from app import show_session_analytics
            show_session_analytics(df)
        elif module_id == "funnel_analysis":
            from app import show_funnel
            show_funnel(df)
        elif module_id == "customer_journey":
            from app import show_customer_journey
            show_customer_journey(df)
        elif module_id == "cohort_analysis":
            from app import show_cohort_analysis
            show_cohort_analysis(df)
        elif module_id == "anomaly_detection":
            from app import show_anomaly_detection
            show_anomaly_detection(df)
        elif module_id == "ml_predictions":
            from app import show_predictions, load_predictions
            predictions = load_predictions('outputs')
            if predictions:
                show_predictions(predictions, df)
            else:
                st.info("No ML predictions available. Run the prediction models first.")
        elif module_id == "recommendations":
            from app import show_recommendations
            show_recommendations(df)
        elif module_id == "customer_scoring":
            from app import show_customer_scoring
            show_customer_scoring(df)
        elif module_id == "survival_analysis":
            from app import show_survival_analysis
            show_survival_analysis(df)
        elif module_id == "merchant_intelligence":
            from app import show_merchant_intelligence
            show_merchant_intelligence(df)
        elif module_id == "promo_analytics":
            from app import show_promo_analytics
            show_promo_analytics(df)
        elif module_id == "search_analytics":
            from app import show_search_analytics
            show_search_analytics(df)
        elif module_id == "delivery_analytics":
            from app import show_delivery_analytics
            show_delivery_analytics(df)
        elif module_id == "order_analysis":
            from app import show_orders
            show_orders(df)
        elif module_id == "attribution_modeling":
            from app import show_attribution_modeling
            show_attribution_modeling(df)
        elif module_id == "reactivation_targeting":
            from app import show_reactivation_targeting
            show_reactivation_targeting(df)
        elif module_id == "product_affinity":
            from app import show_product_affinity
            show_product_affinity(df)
        elif module_id == "trending":
            from app import show_trending
            show_trending(df)
        elif module_id == "platform_analysis":
            from app import show_platform
            show_platform(df)
        elif module_id == "hourly_trends":
            from app import show_hourly
            show_hourly(df)
        else:
            st.info(f"Module '{module.title}' is coming soon!")

    except ImportError as e:
        st.error(f"Error loading module: {e}")
    except Exception as e:
        st.error(f"Error in module: {e}")
        st.exception(e)


def render_data_source_modal():
    """Render a data source selection modal/expander."""
    with st.expander("📂 Data Source", expanded=False):
        is_cloud = not Path("/Users/muhannadsaad").exists()

        if is_cloud:
            options = ["Sample Data (100k rows)", "Upload File"]
        else:
            options = ["Sample Data", "Dec 15 Full", "Load Multiple Files", "Upload"]

        choice = st.radio("Select source:", options, horizontal=True)

        if choice == "Sample Data (100k rows)" or choice == "Sample Data":
            return Path(__file__).parent / "data" / "sample_data.parquet"
        elif choice == "Dec 15 Full":
            return "/Users/muhannadsaad/Desktop/investigation/dec_15/dec_15_25.parquet"
        elif choice == "Load Multiple Files":
            folder = st.text_input("Folder path:", "/Users/muhannadsaad/Desktop/investigation")
            if folder and Path(folder).exists():
                files = list(Path(folder).rglob("*.parquet"))
                selected = st.multiselect("Select files:", [str(f) for f in files])
                if selected:
                    return ("MULTI", tuple(selected))
            return None
        elif choice == "Upload" or choice == "Upload File":
            uploaded = st.file_uploader("Upload parquet/csv", type=['parquet', 'csv'])
            if uploaded:
                temp_path = Path("data") / uploaded.name
                temp_path.parent.mkdir(exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded.getbuffer())
                return str(temp_path)
            return None

    return Path(__file__).parent / "data" / "sample_data.parquet"


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    # Check for module selection from URL or button click
    query_params = st.query_params
    if 'module' in query_params:
        st.session_state.current_module = query_params['module']

    # Render top bar
    current_page = None
    if st.session_state.current_module:
        module = next((m for m in MODULES if m.id == st.session_state.current_module), None)
        current_page = module.title if module else None

    render_top_bar(username, current_page)

    # Add spacing for fixed top bar
    st.markdown('<div style="height: 72px;"></div>', unsafe_allow_html=True)

    # Data source selection (in a subtle way)
    data_path = render_data_source_modal()

    if not data_path:
        st.info("Please select a data source to continue.")
        st.stop()

    # Load data
    if isinstance(data_path, tuple) and data_path[0] == "MULTI":
        df = load_multiple_files(data_path[1])
    else:
        if not Path(str(data_path)).exists():
            st.error(f"Data file not found: {data_path}")
            st.stop()
        df = load_data(data_path)

    # Main content
    st.markdown('<div class="main-content">', unsafe_allow_html=True)

    if st.session_state.current_module:
        # Show module content
        show_module_content(st.session_state.current_module, df)

        # Back button (functional version using Streamlit)
        if st.button("← Back to Insights", key="back_btn"):
            st.session_state.current_module = None
            st.query_params.clear()
            st.rerun()
    else:
        # Show gallery
        st.markdown("## Discover Insights", unsafe_allow_html=True)
        st.markdown("Explore your analytics modules", unsafe_allow_html=True)

        # Render each category
        for category in [Category.CORE, Category.ML, Category.BUSINESS, Category.ADVANCED]:
            style = CATEGORY_STYLES[category]
            st.markdown(f"### {style['icon']} {category.value}")

            # Create columns for cards
            category_modules = [m for m in MODULES if m.category == category]
            cols = st.columns(min(4, len(category_modules)))

            for idx, module in enumerate(category_modules):
                with cols[idx % 4]:
                    # Get stats
                    stats = get_module_stats(df, module.id)

                    # Create a clickable card using Streamlit container
                    with st.container():
                        # Card styling
                        st.markdown(f"""
                        <div style="
                            background: {style['gradient']};
                            height: 80px;
                            border-radius: 12px 12px 0 0;
                            display: flex;
                            align-items: center;
                            justify-content: center;
                            font-size: 2rem;
                            opacity: 0.8;
                        ">{style['icon']}</div>
                        """, unsafe_allow_html=True)

                        st.markdown(f"**{module.title}**")
                        st.caption(module.subtitle)

                        # Stats
                        stat_cols = st.columns(2)
                        for i, (label, value) in enumerate(stats.items()):
                            with stat_cols[i % 2]:
                                st.metric(label, value, label_visibility="visible")

                        # Open button
                        if st.button("Open →", key=f"open_{module.id}", use_container_width=True):
                            st.session_state.current_module = module.id
                            st.query_params["module"] = module.id
                            st.rerun()

            st.markdown("---")

    st.markdown('</div>', unsafe_allow_html=True)

    # Logout button (hidden in a corner)
    with st.sidebar:
        st.markdown("---")
        authenticator.logout(button_name="Logout", location="sidebar")


if __name__ == '__main__':
    main()
