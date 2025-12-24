"""
Snoonu ML Dashboard
===================
Interactive dashboard for consumer analytics and ML insights.

Run with: streamlit run app.py
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
from ui_components import (
    apply_theme, render_logo, render_page_header, render_section_header,
    render_stat_card, render_metric_row, apply_plotly_theme,
    styled_bar_chart, styled_line_chart, styled_pie_chart,
    COLORS, CHART_COLORS, format_number, format_currency, format_percentage,
    render_loading_screen
)
from auth import check_authentication, render_user_menu, log_activity

# Page config
st.set_page_config(
    page_title="Snoonu ML",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom theme
apply_theme()

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


@st.cache_data
def load_data(file_path):
    """Load and cache data."""
    loader = DataLoader()
    return loader.load(file_path)


@st.cache_data
def load_predictions(output_dir):
    """Load prediction outputs if available."""
    predictions = {}
    output_path = Path(output_dir)

    if (output_path / 'churn_predictions_ml.csv').exists():
        predictions['churn'] = pd.read_csv(output_path / 'churn_predictions_ml.csv')
    if (output_path / 'conversion_predictions.csv').exists():
        predictions['conversion'] = pd.read_csv(output_path / 'conversion_predictions.csv')
    if (output_path / 'ltv_predictions.csv').exists():
        predictions['ltv'] = pd.read_csv(output_path / 'ltv_predictions.csv')
    if (output_path / 'churn_feature_importance.csv').exists():
        predictions['churn_importance'] = pd.read_csv(output_path / 'churn_feature_importance.csv')
    if (output_path / 'conversion_feature_importance.csv').exists():
        predictions['conversion_importance'] = pd.read_csv(output_path / 'conversion_feature_importance.csv')
    if (output_path / 'ltv_feature_importance.csv').exists():
        predictions['ltv_importance'] = pd.read_csv(output_path / 'ltv_feature_importance.csv')

    return predictions


@st.cache_data
def build_user_features(df_json):
    """Build and cache user features."""
    df = pd.read_json(df_json)
    engine = FeatureEngine()
    return engine.build_user_features(df)


@st.cache_data
def build_rfm_features(df_json):
    """Build and cache RFM features."""
    df = pd.read_json(df_json)
    engine = FeatureEngine()
    return engine.build_rfm_features(df)


def parse_props(df):
    """Parse event properties."""
    def safe_parse(x):
        if pd.isna(x):
            return {}
        try:
            return json.loads(x)
        except:
            return {}

    if 'event_properties' not in df.columns:
        df = df.copy()
        df['props'] = [{}] * len(df)
        return df

    df = df.copy()
    df['props'] = df['event_properties'].apply(safe_parse)
    return df


def main():
    # Sidebar - Logo
    with st.sidebar:
        render_logo()

        # Data source selection
        st.markdown("##### DATA SOURCE")

        data_option = st.radio(
            "Choose data source:",
            ["Default (Dec 15)", "Dec 9 Data", "Upload File", "Enter Path"],
            label_visibility="collapsed"
        )

        data_path = None

        if data_option == "Default (Dec 15)":
            data_path = "/Users/muhannadsaad/Desktop/investigation/dec_15/dec_15_25.parquet"
        elif data_option == "Dec 9 Data":
            data_path = "/Users/muhannadsaad/Desktop/snoonu-ml-framework/data/amp_data_yesterday.parquet"
        elif data_option == "Upload File":
            uploaded_file = st.file_uploader("Upload parquet/csv", type=['parquet', 'csv'])
            if uploaded_file:
                # Save to temp location
                temp_path = Path("data") / uploaded_file.name
                temp_path.parent.mkdir(exist_ok=True)
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                data_path = str(temp_path)
        else:
            data_path = st.text_input("Enter file path:", label_visibility="collapsed", placeholder="Enter file path...")

    if not data_path or not Path(data_path).exists():
        # Welcome screen
        render_page_header("Snoonu ML", "Analytics & Machine Learning Platform")
        st.info("Select a data source from the sidebar to get started.")
        return

    # Show loading screen while data loads
    loading_placeholder = st.empty()

    # Check if data is already cached
    if 'data_loaded' not in st.session_state or st.session_state.get('current_data_path') != data_path:
        with loading_placeholder.container():
            render_loading_screen()

        df = load_data(data_path)
        predictions = load_predictions('outputs')

        st.session_state.data_loaded = True
        st.session_state.current_data_path = data_path

        # Clear loading screen
        loading_placeholder.empty()
    else:
        df = load_data(data_path)
        predictions = load_predictions('outputs')

    # Sidebar - Analysis selection with icons
    with st.sidebar:
        st.markdown("---")
        st.markdown("##### ANALYTICS")

        # Menu options with icons
        menu_items = [
            ("Overview", "grid"),
            ("Session Analytics", "activity"),
            ("Funnel Analysis", "filter"),
            ("User Segments", "users"),
            ("Cohort Analysis", "layers"),
            ("Recommendations", "star"),
            ("Trending", "trending-up"),
            ("Merchant Intelligence", "briefcase"),
            ("Promo Analytics", "tag"),
            ("Search Analytics", "search"),
            ("Delivery Analytics", "truck"),
            ("Customer Scoring", "award"),
            ("Anomaly Detection", "alert-triangle"),
            ("Attribution Modeling", "git-branch"),
            ("Reactivation Targeting", "refresh-cw"),
            ("Product Affinity", "shopping-bag"),
            ("Order Analysis", "shopping-cart"),
            ("Platform Analysis", "smartphone"),
            ("Hourly Trends", "clock"),
        ]

        # Insert ML Predictions if available
        if predictions:
            menu_items.insert(1, ("ML Predictions", "cpu"))

        analysis_options = [item[0] for item in menu_items]
        analysis_icons = [item[1] for item in menu_items]

        # Try to use streamlit-option-menu for better navigation
        try:
            from streamlit_option_menu import option_menu

            analysis = option_menu(
                menu_title=None,
                options=analysis_options,
                icons=analysis_icons,
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "transparent"},
                    "icon": {"color": COLORS['primary'], "font-size": "14px"},
                    "nav-link": {
                        "font-size": "13px",
                        "text-align": "left",
                        "margin": "2px 0",
                        "padding": "8px 12px",
                        "border-radius": "6px",
                        "color": COLORS['text_secondary'],
                    },
                    "nav-link-selected": {
                        "background-color": COLORS['primary'],
                        "color": "white",
                        "font-weight": "600",
                    },
                }
            )
        except ImportError:
            # Fallback to selectbox
            analysis = st.selectbox(
                "Select analysis:",
                analysis_options,
                label_visibility="collapsed"
            )

        # User menu at bottom of sidebar
        st.markdown("---")
        st.markdown(f"##### ACCOUNT")
        st.markdown(f"Logged in as **{username}**")
        authenticator.logout(button_name="Logout", location="sidebar")

    # Main content based on selection
    if analysis == "Overview":
        show_overview(df)
    elif analysis == "ML Predictions":
        show_predictions(predictions, df)
    elif analysis == "Session Analytics":
        show_session_analytics(df)
    elif analysis == "Funnel Analysis":
        show_funnel(df)
    elif analysis == "User Segments":
        show_segments(df)
    elif analysis == "Cohort Analysis":
        show_cohort_analysis(df)
    elif analysis == "Recommendations":
        show_recommendations(df)
    elif analysis == "Trending":
        show_trending(df)
    elif analysis == "Merchant Intelligence":
        show_merchant_intelligence(df)
    elif analysis == "Promo Analytics":
        show_promo_analytics(df)
    elif analysis == "Search Analytics":
        show_search_analytics(df)
    elif analysis == "Delivery Analytics":
        show_delivery_analytics(df)
    elif analysis == "Customer Scoring":
        show_customer_scoring(df)
    elif analysis == "Anomaly Detection":
        show_anomaly_detection(df)
    elif analysis == "Attribution Modeling":
        show_attribution_modeling(df)
    elif analysis == "Reactivation Targeting":
        show_reactivation_targeting(df)
    elif analysis == "Product Affinity":
        show_product_affinity(df)
    elif analysis == "Order Analysis":
        show_orders(df)
    elif analysis == "Platform Analysis":
        show_platform(df)
    elif analysis == "Hourly Trends":
        show_hourly(df)


def show_overview(df):
    """Show overview dashboard."""
    render_page_header("Overview Dashboard", "Key metrics and insights at a glance")

    # Key metrics
    total_events = len(df)
    unique_users = df['amplitude_id'].nunique()
    homepage_users = df[df['event_type'] == 'homepage_viewed']['amplitude_id'].nunique()
    checkout_users = df[df['event_type'] == 'checkout_completed']['amplitude_id'].nunique()
    total_orders = len(df[df['event_type'] == 'checkout_completed'])

    cvr = checkout_users / homepage_users * 100 if homepage_users > 0 else 0
    opu = total_orders / checkout_users if checkout_users > 0 else 0

    # Metric cards
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Total Events", format_number(total_events))
    with col2:
        st.metric("Unique Users", format_number(unique_users))
    with col3:
        st.metric("Total Orders", format_number(total_orders))
    with col4:
        st.metric("CVR", format_percentage(cvr))
    with col5:
        st.metric("Orders/User", f"{opu:.2f}")

    st.markdown("---")

    # Charts row
    col1, col2 = st.columns(2)

    with col1:
        render_section_header("Event Distribution")
        event_counts = df['event_type'].value_counts().head(10)
        fig = px.bar(
            x=event_counts.values,
            y=event_counts.index,
            orientation='h',
            labels={'x': 'Count', 'y': 'Event Type'},
            color_discrete_sequence=[COLORS['primary']]
        )
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        render_section_header("Platform Distribution")
        platform_counts = df['platform'].value_counts()
        fig = px.pie(
            values=platform_counts.values,
            names=platform_counts.index,
            hole=0.4,
            color_discrete_sequence=CHART_COLORS
        )
        fig.update_layout(height=400)
        fig = apply_plotly_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Orders per user distribution
    render_section_header("Orders per User Distribution")
    orders_by_user = df[df['event_type'] == 'checkout_completed'].groupby('amplitude_id').size()
    order_dist = orders_by_user.value_counts().sort_index().head(10)

    fig = px.bar(
        x=order_dist.index,
        y=order_dist.values,
        labels={'x': 'Orders per User', 'y': 'Number of Users'},
        color_discrete_sequence=[COLORS['chart_2']]
    )
    fig.update_layout(height=300)
    fig = apply_plotly_theme(fig)
    st.plotly_chart(fig, use_container_width=True)


def show_predictions(predictions, df):
    """Show ML prediction results."""
    render_page_header("ML Predictions", "Machine learning model outputs and insights")

    if not predictions:
        st.warning("No prediction outputs found. Run the prediction models first:")
        st.code("""
# Run churn prediction
python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task predict_churn

# Run conversion prediction
python run.py --data data/dec_15.parquet --task predict_conversion

# Run LTV prediction
python run.py --data data/dec_15.parquet --task predict_ltv
        """)
        return

    # Tab selection
    tabs = st.tabs(["Churn", "Conversion", "LTV"])

    with tabs[0]:
        show_churn_predictions(predictions)

    with tabs[1]:
        show_conversion_predictions(predictions)

    with tabs[2]:
        show_ltv_predictions(predictions)


def show_churn_predictions(predictions):
    """Show churn prediction results."""
    if 'churn' not in predictions:
        st.info("Run churn prediction first: `python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task predict_churn`")
        return

    churn_df = predictions['churn'].copy()
    # Handle index column naming
    if 'index' in churn_df.columns and 'amplitude_id' not in churn_df.columns:
        churn_df = churn_df.rename(columns={'index': 'amplitude_id'})

    st.subheader("Churn Prediction Results")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_users = len(churn_df)
    high_risk = len(churn_df[churn_df['churn_probability'] >= 0.7])
    medium_risk = len(churn_df[(churn_df['churn_probability'] >= 0.4) & (churn_df['churn_probability'] < 0.7)])
    low_risk = len(churn_df[churn_df['churn_probability'] < 0.4])

    with col1:
        st.metric("Total Users", f"{total_users:,}")
    with col2:
        st.metric("High Risk (>70%)", f"{high_risk:,}", delta=f"{high_risk/total_users*100:.1f}%")
    with col3:
        st.metric("Medium Risk (40-70%)", f"{medium_risk:,}")
    with col4:
        st.metric("Low Risk (<40%)", f"{low_risk:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Risk distribution
        st.subheader("Churn Probability Distribution")
        fig = px.histogram(churn_df, x='churn_probability', nbins=50,
                          labels={'churn_probability': 'Churn Probability'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Risk segments pie
        st.subheader("Risk Segments")
        risk_data = pd.DataFrame({
            'Risk': ['High Risk', 'Medium Risk', 'Low Risk'],
            'Count': [high_risk, medium_risk, low_risk]
        })
        fig = px.pie(risk_data, values='Count', names='Risk',
                    color='Risk', color_discrete_map={'High Risk': '#ff4444', 'Medium Risk': '#ffaa00', 'Low Risk': '#44ff44'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if 'churn_importance' in predictions:
        st.subheader("Top Churn Predictors")
        importance_df = predictions['churn_importance'].head(10)
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    labels={'importance': 'Feature Importance', 'feature': 'Feature'})
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # High risk users table
    st.subheader("High Risk Users (Top 100)")
    high_risk_users = churn_df.nlargest(100, 'churn_probability')[['amplitude_id', 'churn_probability', 'churn_prediction']]
    high_risk_users['churn_probability'] = (high_risk_users['churn_probability'] * 100).round(1).astype(str) + '%'
    st.dataframe(high_risk_users, use_container_width=True)

    # Download button
    st.download_button(
        "📥 Download All Churn Predictions",
        churn_df.to_csv(index=False),
        "churn_predictions.csv",
        "text/csv"
    )


def show_conversion_predictions(predictions):
    """Show conversion prediction results."""
    if 'conversion' not in predictions:
        st.info("Run conversion prediction first: `python run.py --data data/dec_15.parquet --task predict_conversion`")
        return

    conv_df = predictions['conversion']

    st.subheader("Conversion Prediction Results")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_users = len(conv_df)
    high_intent = len(conv_df[conv_df['conversion_probability'] >= 0.7])
    converted = len(conv_df[conv_df['conversion_prediction'] == 1])
    conversion_rate = converted / total_users * 100

    with col1:
        st.metric("Total Users", f"{total_users:,}")
    with col2:
        st.metric("High Intent (>70%)", f"{high_intent:,}")
    with col3:
        st.metric("Predicted Converters", f"{converted:,}")
    with col4:
        st.metric("Predicted CVR", f"{conversion_rate:.1f}%")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # Probability distribution
        st.subheader("Conversion Probability Distribution")
        fig = px.histogram(conv_df, x='conversion_probability', nbins=50,
                          labels={'conversion_probability': 'Conversion Probability'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Conversion pie
        st.subheader("Predicted Outcomes")
        outcome_data = pd.DataFrame({
            'Outcome': ['Will Convert', 'Will Not Convert'],
            'Count': [converted, total_users - converted]
        })
        fig = px.pie(outcome_data, values='Count', names='Outcome',
                    color='Outcome', color_discrete_map={'Will Convert': '#44ff44', 'Will Not Convert': '#ff4444'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    # Feature importance
    if 'conversion_importance' in predictions:
        st.subheader("Top Conversion Drivers")
        importance_df = predictions['conversion_importance'].head(10)
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    labels={'importance': 'Feature Importance', 'feature': 'Feature'})
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def show_ltv_predictions(predictions):
    """Show LTV prediction results."""
    if 'ltv' not in predictions:
        st.info("Run LTV prediction first: `python run.py --data data/dec_15.parquet --task predict_ltv`")
        return

    ltv_df = predictions['ltv']

    st.subheader("LTV Prediction Results")

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    total_users = len(ltv_df)
    avg_ltv = ltv_df['predicted_ltv'].mean()
    total_ltv = ltv_df['predicted_ltv'].sum()

    with col1:
        st.metric("Total Customers", f"{total_users:,}")
    with col2:
        st.metric("Avg Predicted LTV", f"QAR {avg_ltv:.0f}")
    with col3:
        st.metric("Total Predicted LTV", f"QAR {total_ltv:,.0f}")
    with col4:
        if 'ltv_tier' in ltv_df.columns:
            diamond_users = len(ltv_df[ltv_df['ltv_tier'] == 'Diamond'])
            st.metric("Diamond Tier", f"{diamond_users:,}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        # LTV distribution
        st.subheader("Predicted LTV Distribution")
        fig = px.histogram(ltv_df, x='predicted_ltv', nbins=50,
                          labels={'predicted_ltv': 'Predicted LTV (QAR)'})
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Tier distribution
        if 'ltv_tier' in ltv_df.columns:
            st.subheader("LTV Tier Distribution")
            tier_counts = ltv_df['ltv_tier'].value_counts()
            fig = px.pie(values=tier_counts.values, names=tier_counts.index,
                        color=tier_counts.index,
                        color_discrete_map={'Diamond': '#b9f2ff', 'Gold': '#ffd700', 'Silver': '#c0c0c0', 'Bronze': '#cd7f32'})
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    # Tier summary
    if 'ltv_tier' in ltv_df.columns:
        st.subheader("LTV Tier Summary")
        tier_summary = ltv_df.groupby('ltv_tier').agg({
            'amplitude_id': 'count',
            'predicted_ltv': ['mean', 'sum'],
            'total_revenue': ['mean', 'sum'],
            'order_count': 'mean'
        }).round(2)
        tier_summary.columns = ['Users', 'Avg LTV', 'Total LTV', 'Avg Revenue', 'Total Revenue', 'Avg Orders']
        st.dataframe(tier_summary, use_container_width=True)

    # Feature importance
    if 'ltv_importance' in predictions:
        st.subheader("Top LTV Drivers")
        importance_df = predictions['ltv_importance'].head(10)
        fig = px.bar(importance_df, x='importance', y='feature', orientation='h',
                    labels={'importance': 'Feature Importance', 'feature': 'Feature'})
        fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # High value customers
    st.subheader("Top 100 Highest Value Customers")
    top_customers = ltv_df.nlargest(100, 'predicted_ltv')[['amplitude_id', 'predicted_ltv', 'total_revenue', 'order_count']]
    top_customers['predicted_ltv'] = top_customers['predicted_ltv'].apply(lambda x: f"QAR {x:.0f}")
    top_customers['total_revenue'] = top_customers['total_revenue'].apply(lambda x: f"QAR {x:.0f}")
    st.dataframe(top_customers, use_container_width=True)


def show_funnel(df):
    """Show funnel analysis."""
    render_page_header("Funnel Analysis", "Conversion funnel metrics and drop-off analysis")

    funnel_stages = [
        'homepage_viewed',
        'product_page_viewed',
        'product_added',
        'cart_page_viewed',
        'checkout_button_pressed',
        'payment_initiated',
        'checkout_completed'
    ]

    # Calculate funnel
    funnel_data = []
    for stage in funnel_stages:
        users = df[df['event_type'] == stage]['amplitude_id'].nunique()
        funnel_data.append({'stage': stage.replace('_', ' ').title(), 'users': users})

    funnel_df = pd.DataFrame(funnel_data)

    # Funnel chart
    fig = go.Figure(go.Funnel(
        y=funnel_df['stage'],
        x=funnel_df['users'],
        textinfo="value+percent initial"
    ))
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Stage-to-stage conversion
    st.subheader("Stage-to-Stage Conversion")

    conversion_data = []
    for i in range(len(funnel_stages) - 1):
        current = funnel_stages[i]
        next_stage = funnel_stages[i + 1]

        current_users = set(df[df['event_type'] == current]['amplitude_id'])
        next_users = set(df[df['event_type'] == next_stage]['amplitude_id'])

        continued = len(current_users & next_users)
        cvr = continued / len(current_users) * 100 if current_users else 0

        conversion_data.append({
            'From': current.replace('_', ' ').title(),
            'To': next_stage.replace('_', ' ').title(),
            'Users Continued': continued,
            'Conversion Rate': f"{cvr:.1f}%"
        })

    st.dataframe(pd.DataFrame(conversion_data), use_container_width=True)


def show_segments(df):
    """Show user segments."""
    render_page_header("User Segments", "RFM segmentation and customer profiles")

    with st.spinner("Building RFM segments..."):
        engine = FeatureEngine()
        rfm = engine.build_rfm_features(df)

    if len(rfm) == 0:
        st.warning("No order data found for segmentation")
        return

    # Segment distribution
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Segment Distribution")
        segment_counts = rfm['RFM_segment'].value_counts()
        fig = px.pie(
            values=segment_counts.values,
            names=segment_counts.index,
            hole=0.3
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Segment Sizes")
        fig = px.bar(
            x=segment_counts.index,
            y=segment_counts.values,
            labels={'x': 'Segment', 'y': 'Users'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Segment profiles
    st.subheader("Segment Profiles")
    segment_profiles = rfm.groupby('RFM_segment').agg({
        'recency': 'mean',
        'frequency': 'mean',
        'monetary': 'mean',
        'amplitude_id': 'count'
    }).round(2)
    segment_profiles.columns = ['Avg Recency (days)', 'Avg Frequency', 'Avg Monetary', 'Users']
    segment_profiles = segment_profiles.sort_values('Users', ascending=False)
    st.dataframe(segment_profiles, use_container_width=True)

    # RFM scatter
    st.subheader("RFM Analysis")
    fig = px.scatter(
        rfm,
        x='recency',
        y='frequency',
        size='monetary',
        color='RFM_segment',
        hover_data=['amplitude_id'],
        labels={'recency': 'Recency (days)', 'frequency': 'Frequency', 'monetary': 'Monetary'}
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Download segments
    st.download_button(
        "📥 Download Segments CSV",
        rfm.to_csv(index=False),
        "segments.csv",
        "text/csv"
    )


def show_cohort_analysis(df):
    """Show cohort analysis with segment builder."""
    render_page_header("Cohort Analysis", "Retention, LTV curves, and custom segments")

    from cohort_engine import CohortEngine, SegmentBuilder, PredefinedSegments

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Retention & LTV", "Predefined Segments", "Custom Segment Builder"])

    with tab1:
        show_cohort_retention(df)

    with tab2:
        show_predefined_segments(df)

    with tab3:
        show_segment_builder(df)


def show_cohort_retention(df):
    """Show retention heatmap and LTV curves."""
    from cohort_engine import CohortEngine

    st.subheader("Cohort Retention Analysis")

    # Cohort type selector
    cohort_type = st.selectbox(
        "Cohort By:",
        ["first_order_week", "first_order_month", "platform", "first_aov_tier"],
        format_func=lambda x: {
            "first_order_week": "First Order Week",
            "first_order_month": "First Order Month",
            "platform": "Platform (iOS/Android)",
            "first_aov_tier": "First Order AOV Tier"
        }.get(x, x)
    )

    engine = CohortEngine()

    with st.spinner("Building cohorts..."):
        cohorts = engine.build_cohorts(df, cohort_type=cohort_type)

    if len(cohorts) == 0:
        st.warning("No order data found for cohort analysis")
        return

    # Cohort summary
    summary = engine.get_cohort_summary(df, cohorts)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Cohorts", f"{cohorts['cohort'].nunique()}")
    with col2:
        st.metric("Total Users", f"{len(cohorts):,}")
    with col3:
        st.metric("Avg LTV", f"QAR {summary['ltv'].mean():.0f}")

    # Cohort summary table
    st.subheader("Cohort Summary")
    st.dataframe(summary.style.format({
        'users': '{:,}',
        'orders': '{:,}',
        'total_revenue': 'QAR {:,.0f}',
        'avg_order_value': 'QAR {:.0f}',
        'orders_per_user': '{:.2f}',
        'ltv': 'QAR {:.0f}'
    }), use_container_width=True)

    # Retention heatmap
    st.subheader("Retention Heatmap")
    with st.spinner("Calculating retention..."):
        retention = engine.calculate_retention(df, cohorts, period='week', max_periods=8)

    if len(retention) > 0:
        fig = px.imshow(
            retention.values,
            labels=dict(x="Week", y="Cohort", color="Retention %"),
            x=[f"W{i}" for i in range(retention.shape[1])],
            y=retention.index.tolist(),
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        fig.update_layout(height=max(300, len(retention) * 40))
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Retention data requires multiple weeks of data")

    # Time to 2nd order
    st.subheader("Time to 2nd Order")
    time_to_2nd = engine.calculate_time_to_nth_order(df, n=2)

    if len(time_to_2nd) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Users with 2+ Orders", f"{len(time_to_2nd):,}")
        with col2:
            st.metric("Median Days", f"{time_to_2nd['days_to_nth_order'].median():.1f}")
        with col3:
            st.metric("Mean Days", f"{time_to_2nd['days_to_nth_order'].mean():.1f}")
        with col4:
            st.metric("75th Percentile", f"{time_to_2nd['days_to_nth_order'].quantile(0.75):.1f}")

        # Distribution
        fig = px.histogram(time_to_2nd, x='days_to_nth_order', nbins=30,
                          labels={'days_to_nth_order': 'Days to 2nd Order'})
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # Download cohort data
    st.download_button(
        "📥 Download Cohort Assignments",
        cohorts.to_csv(index=False),
        "cohort_assignments.csv",
        "text/csv"
    )


def show_predefined_segments(df):
    """Show predefined segment counts and export."""
    from cohort_engine import PredefinedSegments

    st.subheader("Predefined User Segments")
    st.markdown("Click on any segment to export user IDs for campaigns, retargeting, etc.")

    # Define all predefined segments
    segments_def = [
        ("Cart Abandoners", "Users who added to cart but didn't checkout", PredefinedSegments.cart_abandoners),
        ("One-Time Buyers", "Users who ordered exactly once", PredefinedSegments.one_time_buyers),
        ("Repeat Buyers", "Users who ordered 2+ times", PredefinedSegments.repeat_buyers),
        ("Power Users", "Users who ordered 5+ times", PredefinedSegments.power_users),
        ("Browsers (Never Bought)", "Users who viewed products but never ordered", PredefinedSegments.browsers_not_buyers),
    ]

    # Calculate all segments
    segment_data = []
    for name, desc, func in segments_def:
        segment = func(df)
        segment_data.append({
            'name': name,
            'description': desc,
            'count': segment.count(),
            'segment': segment
        })

    # Display in columns
    for i in range(0, len(segment_data), 2):
        cols = st.columns(2)
        for j, col in enumerate(cols):
            if i + j < len(segment_data):
                seg = segment_data[i + j]
                with col:
                    st.markdown(f"### {seg['name']}")
                    st.markdown(f"*{seg['description']}*")
                    st.metric("Users", f"{seg['count']:,}")

                    # Export button
                    export_df = seg['segment'].export()
                    st.download_button(
                        f"📥 Export {seg['name']}",
                        export_df.to_csv(index=False),
                        f"{seg['name'].lower().replace(' ', '_')}.csv",
                        "text/csv",
                        key=f"export_{i+j}"
                    )

    # Segment comparison chart
    st.subheader("Segment Size Comparison")
    comparison_df = pd.DataFrame([
        {'Segment': s['name'], 'Users': s['count']}
        for s in segment_data
    ])
    fig = px.bar(comparison_df, x='Segment', y='Users')
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def show_segment_builder(df):
    """Interactive segment builder with custom filters."""
    from cohort_engine import SegmentBuilder

    st.subheader("Custom Segment Builder")
    st.markdown("Build custom segments by combining multiple filters. Export user IDs for targeting.")

    # Initialize segment builder
    segment = SegmentBuilder(df)
    all_users = segment.count()

    st.info(f"Starting with **{all_users:,}** total users")

    # Filter options
    st.markdown("### Add Filters")

    col1, col2 = st.columns(2)

    with col1:
        # Event-based filters
        st.markdown("**Event Filters**")

        did_checkout = st.checkbox("Placed at least 1 order")
        if did_checkout:
            segment = segment.did_event('checkout_completed')

        never_ordered = st.checkbox("Never ordered")
        if never_ordered:
            segment = segment.never_ordered()

        added_to_cart = st.checkbox("Added to cart")
        if added_to_cart:
            segment = segment.did_event('product_added')

        viewed_products = st.checkbox("Viewed products")
        if viewed_products:
            segment = segment.did_event('product_page_viewed')

    with col2:
        # Order count filters
        st.markdown("**Order Count**")

        order_filter = st.selectbox(
            "Order count filter:",
            ["No filter", "Exactly 1 order", "2+ orders", "5+ orders", "Less than 3 orders"]
        )

        if order_filter == "Exactly 1 order":
            segment = segment.ordered_exactly(1)
        elif order_filter == "2+ orders":
            segment = segment.ordered_at_least(2)
        elif order_filter == "5+ orders":
            segment = segment.ordered_at_least(5)
        elif order_filter == "Less than 3 orders":
            segment = segment.ordered_less_than(3)

        # Platform filter
        st.markdown("**Platform**")
        platform_filter = st.selectbox(
            "Platform:",
            ["All", "iOS", "Android"]
        )
        if platform_filter != "All":
            segment = segment.platform(platform_filter.lower())

    # Advanced filters
    with st.expander("Advanced Filters"):
        col1, col2 = st.columns(2)

        with col1:
            # Recency filters
            active_days = st.number_input("Active in last N days (0 = no filter)", min_value=0, value=0)
            if active_days > 0:
                segment = segment.active_in_last_n_days(active_days)

            inactive_days = st.number_input("Inactive for N+ days (0 = no filter)", min_value=0, value=0)
            if inactive_days > 0:
                segment = segment.inactive_for_n_days(inactive_days)

        with col2:
            # ML-based filters (if predictions exist)
            use_churn = st.checkbox("High churn risk (>70%)")
            if use_churn:
                segment = segment.with_churn_risk(0.7)

            ltv_tier = st.selectbox("LTV Tier", ["All", "Diamond", "Gold", "Silver", "Bronze"])
            if ltv_tier != "All":
                segment = segment.with_ltv_tier(ltv_tier)

    # Results
    st.markdown("---")
    st.markdown("### Segment Results")

    result_count = segment.count()
    pct = result_count / all_users * 100 if all_users > 0 else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Users in Segment", f"{result_count:,}")
    with col2:
        st.metric("% of Total", f"{pct:.1f}%")
    with col3:
        st.metric("Filters Applied", len(segment.filters_applied))

    # Show applied filters
    if segment.filters_applied:
        st.markdown("**Applied Filters:**")
        for f in segment.filters_applied:
            st.markdown(f"- {f}")

    # Export button
    if result_count > 0:
        export_df = segment.export()
        st.download_button(
            "📥 Export Segment (CSV)",
            export_df.to_csv(index=False),
            "custom_segment.csv",
            "text/csv",
            key="export_custom"
        )

        # Preview
        with st.expander("Preview User IDs"):
            st.dataframe(export_df.head(100), use_container_width=True)
    else:
        st.warning("No users match the selected filters")


def show_recommendations(df):
    """Show recommendation engine interface."""
    render_page_header("Recommendations", "Personalized merchant recommendations")

    # Check for existing model and recommendations
    model_path = Path('outputs/recommendations/item_item_model.pkl')
    recs_path = Path('outputs/recommendations/user_recommendations.parquet')
    metrics_path = Path('outputs/recommendations/model_metrics.json')

    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Model Status", "User Lookup", "Merchant Analysis", "CRM Segments"])

    with tab1:
        show_rec_model_status(df, model_path, metrics_path, recs_path)

    with tab2:
        show_user_recommendations(df, model_path, recs_path)

    with tab3:
        show_merchant_analysis(df, model_path)

    with tab4:
        show_rec_segments(df, model_path)


def show_rec_model_status(df, model_path, metrics_path, recs_path):
    """Show recommendation model status and metrics."""
    st.subheader("Model Status")

    if not model_path.exists():
        st.warning("No trained recommendation model found.")
        st.markdown("**Train the model using:**")
        st.code("""
# Train and evaluate with temporal split:
python run.py --data data/dec_9.parquet --future data/dec_15.parquet --task recommend

# Or train on single dataset:
python run.py --data data/dec_15.parquet --task recommend
        """)

        # Option to train from dashboard
        if st.button("Train Model Now"):
            with st.spinner("Training recommendation model..."):
                try:
                    from recommendations import ItemItemRecommender
                    recommender = ItemItemRecommender(min_support=3, n_neighbors=20)
                    recommender.fit(df)

                    # Save model
                    import pickle
                    model_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(model_path, 'wb') as f:
                        pickle.dump(recommender, f)

                    # Generate recommendations for all users
                    all_users = list(recommender.user_to_idx.keys())
                    all_recs = recommender.recommend_batch(all_users, n=10)
                    all_recs.to_parquet(recs_path, index=False)

                    st.success("Model trained successfully! Refresh the page to see results.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error training model: {e}")
        return

    # Model exists - show stats
    st.success("Model is trained and ready")

    import pickle
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)

    stats = recommender.get_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Users", f"{stats['n_users']:,}")
    with col2:
        st.metric("Merchants", f"{stats['n_merchants']:,}")
    with col3:
        st.metric("Avg Orders/User", f"{stats['avg_orders_per_user']:.2f}")
    with col4:
        st.metric("Matrix Density", f"{stats['matrix_density']:.4%}")

    # Show evaluation metrics if available
    if metrics_path.exists():
        st.subheader("Model Performance")
        import json
        with open(metrics_path) as f:
            metrics = json.load(f)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**@5 Metrics**")
            st.metric("Precision@5", f"{metrics.get('mean_precision@5', 0):.4f}")
            st.metric("Hit Rate@5", f"{metrics.get('mean_hit_rate@5', 0):.4f}")

        with col2:
            st.markdown("**@10 Metrics**")
            st.metric("Precision@10", f"{metrics.get('mean_precision@10', 0):.4f}")
            st.metric("Hit Rate@10", f"{metrics.get('mean_hit_rate@10', 0):.4f}")

        with col3:
            st.markdown("**Overall**")
            st.metric("Coverage", f"{metrics.get('coverage', 0):.4f}")
            if 'mean_diversity' in metrics:
                st.metric("Diversity", f"{metrics['mean_diversity']:.4f}")

        # Metrics explanation
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            - **Precision@K**: Of top K recommendations, what fraction did the user actually order?
            - **Hit Rate@K**: Did at least one of the top K recommendations get ordered?
            - **Coverage**: What fraction of merchants ever get recommended?
            - **Diversity**: How different are the recommendations from each other?
            """)

    # Show recommendation stats
    if recs_path.exists():
        st.subheader("Recommendation Stats")
        recs_df = pd.read_parquet(recs_path)
        st.info(f"Generated {len(recs_df):,} recommendations for {recs_df['amplitude_id'].nunique():,} users")


def show_user_recommendations(df, model_path, recs_path):
    """Look up recommendations for a specific user."""
    st.subheader("User Recommendation Lookup")

    if not model_path.exists():
        st.info("Train the model first in the Model Status tab")
        return

    import pickle
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)

    # User ID input
    user_id_input = st.text_input("Enter User ID (amplitude_id):")

    if user_id_input:
        try:
            user_id = int(user_id_input)
        except ValueError:
            st.error("Please enter a valid numeric user ID")
            return

        if user_id not in recommender.user_to_idx:
            st.warning(f"User {user_id} not found in training data. This could be a new user.")
            return

        # Get recommendations
        recs = recommender.recommend(user_id, n=10, exclude_ordered=True)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### Recommendations")
            if recs:
                rec_df = pd.DataFrame([
                    {
                        'Rank': i+1,
                        'Merchant': recommender.merchant_names.get(m, 'Unknown'),
                        'Merchant ID': m,
                        'Score': f"{s:.3f}"
                    }
                    for i, (m, s) in enumerate(recs)
                ])
                st.dataframe(rec_df, use_container_width=True, hide_index=True)
            else:
                st.info("No recommendations available for this user")

        with col2:
            st.markdown("### Order History")
            # Get user's order history
            user_orders = df[(df['amplitude_id'] == user_id) & (df['event_type'] == 'checkout_completed')]

            if len(user_orders) > 0:
                user_orders = parse_props(user_orders)
                merchants = user_orders['props'].apply(lambda x: x.get('merchant_name', 'Unknown'))
                order_counts = merchants.value_counts()

                history_df = pd.DataFrame({
                    'Merchant': order_counts.index,
                    'Orders': order_counts.values
                })
                st.dataframe(history_df, use_container_width=True, hide_index=True)
            else:
                st.info("No order history found")

        # Explain recommendations
        if recs:
            with st.expander("Why these recommendations?"):
                for merchant_id, score in recs[:3]:
                    explanation = recommender.explain_recommendation(user_id, merchant_id)
                    st.markdown(f"**{recommender.merchant_names.get(merchant_id, 'Unknown')}**")
                    if 'based_on' in explanation and explanation['based_on']:
                        st.markdown("Recommended because you ordered from:")
                        for contrib in explanation['based_on'][:3]:
                            st.markdown(f"  - {contrib['merchant_name']} (similarity: {contrib['similarity']:.3f})")
                    st.markdown("---")


def show_merchant_analysis(df, model_path):
    """Analyze a specific merchant's potential customers."""
    st.subheader("Merchant Analysis")

    if not model_path.exists():
        st.info("Train the model first in the Model Status tab")
        return

    import pickle
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)

    # Get merchant list
    merchants = [(m, recommender.merchant_names.get(m, 'Unknown'))
                 for m in recommender.merchant_to_idx.keys()]
    merchants.sort(key=lambda x: x[1])

    # Merchant selector
    merchant_options = {f"{name} ({mid})": mid for mid, name in merchants}
    selected = st.selectbox("Select Merchant:", list(merchant_options.keys()))

    if selected:
        merchant_id = merchant_options[selected]
        merchant_name = recommender.merchant_names.get(merchant_id, 'Unknown')

        st.markdown(f"### {merchant_name}")

        col1, col2 = st.columns(2)

        with col1:
            # Similar merchants
            st.markdown("**Similar Merchants**")
            similar = recommender.get_similar_merchants(merchant_id, n=10)

            if similar:
                similar_df = pd.DataFrame([
                    {
                        'Merchant': recommender.merchant_names.get(m, 'Unknown'),
                        'Similarity': f"{s:.3f}"
                    }
                    for m, s in similar
                ])
                st.dataframe(similar_df, use_container_width=True, hide_index=True)
            else:
                st.info("No similar merchants found")

        with col2:
            # Current stats
            st.markdown("**Current Performance**")
            merchant_orders = df[(df['event_type'] == 'checkout_completed')]
            merchant_orders = parse_props(merchant_orders)
            merchant_orders['merchant_id'] = merchant_orders['props'].apply(lambda x: x.get('merchant_id'))

            this_merchant = merchant_orders[merchant_orders['merchant_id'] == merchant_id]

            st.metric("Total Orders", f"{len(this_merchant):,}")
            st.metric("Unique Customers", f"{this_merchant['amplitude_id'].nunique():,}")

        # Find high affinity prospects
        st.markdown("### High Affinity Prospects")
        st.markdown("Users who haven't ordered but have high predicted affinity")

        if st.button("Find Prospects", key="find_prospects"):
            with st.spinner("Analyzing user affinities..."):
                from recommendations import RecommendationSegments
                segments = RecommendationSegments(recommender, df)
                prospects = segments.high_affinity_prospects(merchant_id, min_score=0.3, max_users=100)

                if len(prospects) > 0:
                    st.success(f"Found {len(prospects)} high-affinity prospects")
                    st.dataframe(prospects, use_container_width=True)

                    st.download_button(
                        "📥 Export Prospects",
                        prospects.to_csv(index=False),
                        f"prospects_{merchant_id}.csv",
                        "text/csv"
                    )
                else:
                    st.info("No high-affinity prospects found for this merchant")


def show_rec_segments(df, model_path):
    """Show recommendation-based CRM segments."""
    st.subheader("CRM Segments with Recommendations")

    if not model_path.exists():
        st.info("Train the model first in the Model Status tab")
        return

    import pickle
    with open(model_path, 'rb') as f:
        recommender = pickle.load(f)

    from recommendations import RecommendationSegments
    segments = RecommendationSegments(recommender, df)

    # Segment type selector
    segment_type = st.selectbox(
        "Select Segment Type:",
        [
            "Lapsed Users with Recommendations",
            "New User Onboarding",
            "Cross-sell Targets",
            "Merchant Superfans"
        ]
    )

    if segment_type == "Lapsed Users with Recommendations":
        st.markdown("Users who haven't ordered recently, with personalized recommendations to bring them back")

        col1, col2 = st.columns(2)
        with col1:
            min_days = st.number_input("Min days inactive", value=7, min_value=1)
        with col2:
            max_days = st.number_input("Max days inactive", value=30, min_value=1)

        if st.button("Generate Segment", key="gen_lapsed"):
            with st.spinner("Building segment..."):
                result = segments.lapsed_with_recommendations(min_days, max_days)
                st.success(f"Found {len(result):,} lapsed users")

                if len(result) > 0:
                    st.dataframe(result.head(100), use_container_width=True)
                    st.download_button(
                        "📥 Export Full Segment",
                        result.to_csv(index=False),
                        "lapsed_with_recs.csv",
                        "text/csv"
                    )

    elif segment_type == "New User Onboarding":
        st.markdown("New users with limited orders, with discovery recommendations")

        max_orders = st.number_input("Max orders to qualify as 'new'", value=2, min_value=1)

        if st.button("Generate Segment", key="gen_new"):
            with st.spinner("Building segment..."):
                result = segments.new_user_onboarding(max_orders=max_orders)
                st.success(f"Found {len(result):,} new users")

                if len(result) > 0:
                    st.dataframe(result.head(100), use_container_width=True)
                    st.download_button(
                        "📥 Export Full Segment",
                        result.to_csv(index=False),
                        "new_user_onboarding.csv",
                        "text/csv"
                    )

    elif segment_type == "Cross-sell Targets":
        st.markdown("Users who ordered from one merchant and would likely enjoy another")

        merchants = [(m, recommender.merchant_names.get(m, 'Unknown'))
                     for m in recommender.merchant_to_idx.keys()]
        merchants.sort(key=lambda x: x[1])
        merchant_options = {f"{name}": mid for mid, name in merchants}

        col1, col2 = st.columns(2)
        with col1:
            source = st.selectbox("Source Merchant (users ordered from):", list(merchant_options.keys()), key="source_m")
        with col2:
            target = st.selectbox("Target Merchant (to recommend):", list(merchant_options.keys()), key="target_m")

        if st.button("Find Cross-sell Targets", key="gen_cross"):
            if source and target:
                with st.spinner("Analyzing cross-sell potential..."):
                    result = segments.cross_sell_targets(
                        merchant_options[source],
                        merchant_options[target],
                        min_score=0.2
                    )
                    st.success(f"Found {len(result):,} cross-sell targets")

                    if len(result) > 0:
                        st.dataframe(result.head(100), use_container_width=True)
                        st.download_button(
                            "📥 Export Cross-sell Targets",
                            result.to_csv(index=False),
                            "cross_sell_targets.csv",
                            "text/csv"
                        )

    elif segment_type == "Merchant Superfans":
        st.markdown("Top customers for a specific merchant")

        merchants = [(m, recommender.merchant_names.get(m, 'Unknown'))
                     for m in recommender.merchant_to_idx.keys()]
        merchants.sort(key=lambda x: x[1])
        merchant_options = {f"{name}": mid for mid, name in merchants}

        selected_merchant = st.selectbox("Select Merchant:", list(merchant_options.keys()), key="superfan_m")
        top_n = st.number_input("Top N users", value=100, min_value=10, max_value=1000)

        if st.button("Find Superfans", key="gen_superfans"):
            with st.spinner("Finding superfans..."):
                result = segments.merchant_superfans(merchant_options[selected_merchant], top_n=top_n)

                if len(result) > 0:
                    st.success(f"Found {len(result):,} superfans")
                    st.dataframe(result, use_container_width=True)
                    st.download_button(
                        "📥 Export Superfans",
                        result.to_csv(index=False),
                        "merchant_superfans.csv",
                        "text/csv"
                    )
                else:
                    st.warning("No orders found for this merchant")


def show_trending(df):
    """Show trending and popularity features."""
    render_page_header("Trending", "Popular merchants and trending patterns")

    from recommendations import TrendingEngine

    with st.spinner("Analyzing trending data..."):
        try:
            engine = TrendingEngine(df)
        except Exception as e:
            st.error(f"Error initializing trending engine: {e}")
            st.info("Make sure the dataset contains checkout_completed events")
            return

    # Summary stats
    summary = engine.get_summary()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Orders", f"{summary['total_orders']:,}")
    with col2:
        st.metric("Unique Merchants", f"{summary['unique_merchants']:,}")
    with col3:
        st.metric("Unique Customers", f"{summary['unique_customers']:,}")
    with col4:
        st.metric("Avg Orders/Merchant", f"{summary['avg_orders_per_merchant']:.1f}")

    st.markdown("---")

    # Tabs for different views
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Popular Now",
        "By Time Slot",
        "By Platform",
        "Customer Insights",
        "Contextual Recs"
    ])

    with tab1:
        show_popular_now(engine)

    with tab2:
        show_time_based_popularity(engine)

    with tab3:
        show_platform_popularity(engine)

    with tab4:
        show_customer_insights(engine)

    with tab5:
        show_contextual_recommendations(engine)


def show_popular_now(engine):
    """Show currently popular merchants."""
    st.subheader("Popular Merchants")

    col1, col2 = st.columns([2, 1])

    with col2:
        n_merchants = st.slider("Number of merchants", 5, 50, 20, key="pop_n")
        min_orders = st.slider("Minimum orders", 1, 20, 5, key="pop_min")

    popular = engine.popular_now(n=n_merchants, min_orders=min_orders)

    with col1:
        if len(popular) > 0:
            # Bar chart
            fig = px.bar(
                popular.head(15),
                x='order_count',
                y='merchant',
                orientation='h',
                labels={'order_count': 'Orders', 'merchant': 'Merchant'},
                color='orders_per_customer',
                color_continuous_scale='Blues',
                title='Top Merchants by Order Count'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No merchants meet the minimum order threshold")

    # Full table
    if len(popular) > 0:
        st.markdown("### Full Rankings")
        st.dataframe(
            popular.style.format({
                'order_count': '{:,}',
                'unique_customers': '{:,}',
                'orders_per_customer': '{:.2f}'
            }),
            use_container_width=True
        )

        st.download_button(
            "📥 Export Popular Merchants",
            popular.to_csv(index=False),
            "popular_merchants.csv",
            "text/csv"
        )

    # Trending velocity
    st.markdown("---")
    st.subheader("Trending by Velocity")
    st.markdown("Merchants with highest orders per hour (hot right now)")

    velocity = engine.trending_velocity(n=10, min_orders=3)
    if len(velocity) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                velocity.style.format({
                    'order_count': '{:,}',
                    'orders_per_hour': '{:.2f}'
                }),
                use_container_width=True
            )
        with col2:
            fig = px.bar(
                velocity,
                x='orders_per_hour',
                y='merchant',
                orientation='h',
                labels={'orders_per_hour': 'Orders/Hour', 'merchant': 'Merchant'},
                color='orders_per_hour',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'}, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)


def show_time_based_popularity(engine):
    """Show popularity by time slot."""
    st.subheader("Popular by Time of Day")

    time_slots = ['breakfast', 'lunch', 'afternoon', 'dinner', 'late_night']
    time_labels = {
        'breakfast': '🌅 Breakfast (6-11)',
        'lunch': '☀️ Lunch (11-15)',
        'afternoon': '🌤️ Afternoon (15-18)',
        'dinner': '🌙 Dinner (18-22)',
        'late_night': '🌃 Late Night (22-6)'
    }

    selected_slot = st.selectbox(
        "Select Time Slot:",
        time_slots,
        format_func=lambda x: time_labels.get(x, x)
    )

    popular = engine.popular_by_time(selected_slot, n=15)

    if len(popular) > 0:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.bar(
                popular,
                x='order_count',
                y='merchant',
                orientation='h',
                labels={'order_count': 'Orders', 'merchant': 'Merchant'},
                title=f'Top Merchants: {time_labels[selected_slot]}'
            )
            fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(
                popular[['rank', 'merchant', 'order_count', 'unique_customers']].style.format({
                    'order_count': '{:,}',
                    'unique_customers': '{:,}'
                }),
                use_container_width=True,
                hide_index=True
            )

        st.download_button(
            f"📥 Export {selected_slot.title()} Popular",
            popular.to_csv(index=False),
            f"popular_{selected_slot}.csv",
            "text/csv"
        )
    else:
        st.info(f"No order data available for {selected_slot}")

    # Weekend vs Weekday comparison
    st.markdown("---")
    st.subheader("Weekend vs Weekday")

    weekend_weekday = engine.weekend_vs_weekday(n=10)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Weekend (Fri-Sat)**")
        if len(weekend_weekday['weekend']) > 0:
            st.dataframe(
                weekend_weekday['weekend'][['rank', 'merchant', 'order_count']].style.format({
                    'order_count': '{:,}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No weekend data")

    with col2:
        st.markdown("**Weekday (Sun-Thu)**")
        if len(weekend_weekday['weekday']) > 0:
            st.dataframe(
                weekend_weekday['weekday'][['rank', 'merchant', 'order_count']].style.format({
                    'order_count': '{:,}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No weekday data")


def show_platform_popularity(engine):
    """Show popularity by platform."""
    st.subheader("Popular by Platform")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🍎 iOS")
        ios_pop = engine.popular_by_platform('ios', n=10)
        if len(ios_pop) > 0:
            fig = px.bar(
                ios_pop,
                x='order_count',
                y='merchant',
                orientation='h',
                labels={'order_count': 'Orders', 'merchant': 'Merchant'}
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No iOS data available")

    with col2:
        st.markdown("### 🤖 Android")
        android_pop = engine.popular_by_platform('android', n=10)
        if len(android_pop) > 0:
            fig = px.bar(
                android_pop,
                x='order_count',
                y='merchant',
                orientation='h',
                labels={'order_count': 'Orders', 'merchant': 'Merchant'},
                color_discrete_sequence=['#3DDC84']  # Android green
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No Android data available")

    # Platform comparison table
    st.markdown("---")
    st.subheader("Platform Comparison")

    if len(ios_pop) > 0 and len(android_pop) > 0:
        ios_merchants = set(ios_pop['merchant'].head(10))
        android_merchants = set(android_pop['merchant'].head(10))

        both = ios_merchants & android_merchants
        ios_only = ios_merchants - android_merchants
        android_only = android_merchants - ios_merchants

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Top 10 on Both", len(both))
        with col2:
            st.metric("iOS-only Top 10", len(ios_only))
        with col3:
            st.metric("Android-only Top 10", len(android_only))

        if ios_only:
            st.markdown(f"**iOS favorites (not in Android top 10):** {', '.join(list(ios_only)[:5])}")
        if android_only:
            st.markdown(f"**Android favorites (not in iOS top 10):** {', '.join(list(android_only)[:5])}")


def show_customer_insights(engine):
    """Show new vs repeat customer insights."""
    st.subheader("Customer Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 👋 New Customer Favorites")
        st.markdown("Popular with first-time orderers (great for onboarding)")

        new_faves = engine.new_customer_favorites(n=10)
        if len(new_faves) > 0:
            fig = px.bar(
                new_faves,
                x='new_customer_orders',
                y='merchant',
                orientation='h',
                labels={'new_customer_orders': 'New Customer Orders', 'merchant': 'Merchant'},
                color='new_customer_pct',
                color_continuous_scale='Greens',
                hover_data=['order_count', 'new_customer_pct']
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                new_faves[['rank', 'merchant', 'new_customer_orders', 'order_count', 'new_customer_pct']].style.format({
                    'new_customer_orders': '{:,}',
                    'order_count': '{:,}',
                    'new_customer_pct': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Insufficient data for new customer analysis")

    with col2:
        st.markdown("### 🔄 Repeat Customer Favorites")
        st.markdown("Highest repeat order rates (quality indicator)")

        repeat_faves = engine.repeat_customer_favorites(n=10)
        if len(repeat_faves) > 0:
            fig = px.bar(
                repeat_faves,
                x='repeat_rate',
                y='merchant',
                orientation='h',
                labels={'repeat_rate': 'Repeat Rate (%)', 'merchant': 'Merchant'},
                color='repeat_rate',
                color_continuous_scale='Blues',
                hover_data=['total_customers', 'repeat_customers']
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                repeat_faves[['rank', 'merchant', 'total_customers', 'repeat_customers', 'repeat_rate']].style.format({
                    'total_customers': '{:,}',
                    'repeat_customers': '{:,}',
                    'repeat_rate': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Insufficient data for repeat customer analysis")

    # Category breakdown
    st.markdown("---")
    st.subheader("Category Breakdown")

    categories = engine.category_breakdown()
    if len(categories) > 0:
        col1, col2 = st.columns(2)

        with col1:
            fig = px.pie(
                categories.head(8),
                values='order_count',
                names='category',
                title='Order Distribution by Category',
                hole=0.4
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.dataframe(
                categories.style.format({
                    'order_count': '{:,}',
                    'unique_customers': '{:,}',
                    'merchant_count': '{:,}',
                    'pct_of_orders': '{:.1f}%'
                }),
                use_container_width=True
            )
    else:
        st.info("Category data not available in this dataset")


def show_contextual_recommendations(engine):
    """Show contextual recommendations based on current context."""
    st.subheader("Contextual Recommendations")
    st.markdown("Get recommendations based on current time, platform, and user type")

    col1, col2, col3 = st.columns(3)

    with col1:
        hour = st.slider("Hour of day", 0, 23, datetime.now().hour)

    with col2:
        platform = st.selectbox("Platform", ["Any", "iOS", "Android"])
        platform_val = None if platform == "Any" else platform.lower()

    with col3:
        is_new_user = st.checkbox("New user?", value=False)

    n_recs = st.slider("Number of recommendations", 5, 20, 10)

    if st.button("Get Recommendations", key="get_context_recs"):
        recs = engine.get_recommendations_for_context(
            hour=hour,
            platform=platform_val,
            is_new_user=is_new_user,
            n=n_recs
        )

        if len(recs) > 0:
            st.success(f"Generated {len(recs)} contextual recommendations")

            # Show context info
            st.info(f"Context: {recs['context'].iloc[0]}")

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    recs,
                    x='order_count',
                    y='merchant',
                    orientation='h',
                    labels={'order_count': 'Orders', 'merchant': 'Merchant'},
                    title='Recommended Merchants'
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    recs[['rank', 'merchant', 'order_count']].style.format({
                        'order_count': '{:,}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            st.download_button(
                "📥 Export Recommendations",
                recs.to_csv(index=False),
                "contextual_recommendations.csv",
                "text/csv"
            )
        else:
            st.warning("No recommendations available for this context")


def show_session_analytics(df):
    """Show session and funnel analytics."""
    render_page_header("Session Analytics", "Session behavior and funnel deep-dive")

    from analytics import SessionAnalyzer, FunnelAnalyzer, PathAnalyzer

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Session Analysis", "Funnel Deep Dive", "User Journeys"])

    with tab1:
        show_session_analysis(df)

    with tab2:
        show_funnel_deep_dive(df)

    with tab3:
        show_path_analysis(df)


def show_session_analysis(df):
    """Show session metrics and patterns."""
    from analytics import SessionAnalyzer

    with st.spinner("Building sessions..."):
        try:
            analyzer = SessionAnalyzer(df)
        except Exception as e:
            st.error(f"Error analyzing sessions: {e}")
            return

    # Summary metrics
    summary = analyzer.get_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Sessions", f"{summary['total_sessions']:,}")
    with col2:
        st.metric("Bounce Rate", f"{summary['bounce_rate']:.1f}%")
    with col3:
        st.metric("Session CVR", f"{summary['conversion_rate']:.1f}%")
    with col4:
        st.metric("Avg Duration", f"{summary['avg_session_duration']:.1f} min")
    with col5:
        st.metric("Avg Events", f"{summary['avg_events_per_session']:.1f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Session Duration Distribution")
        duration_dist = analyzer.session_duration_distribution()
        if len(duration_dist) > 0:
            fig = px.bar(
                duration_dist,
                x='duration_bucket',
                y='session_count',
                color='conversion_rate',
                color_continuous_scale='Greens',
                labels={'duration_bucket': 'Duration', 'session_count': 'Sessions', 'conversion_rate': 'CVR %'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Session Depth vs Conversion")
        depth_analysis = analyzer.get_session_depth_analysis()
        if len(depth_analysis) > 0:
            fig = px.bar(
                depth_analysis,
                x='session_depth',
                y='conversion_rate',
                labels={'session_depth': 'Events in Session', 'conversion_rate': 'Conversion Rate %'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Bounce Rate by Entry Point")
        bounce_by_entry = analyzer.bounce_rate_by_entry()
        if len(bounce_by_entry) > 0:
            st.dataframe(
                bounce_by_entry.head(10).style.format({
                    'sessions': '{:,}',
                    'bounce_rate': '{:.1f}%',
                    'conversion_rate': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

    with col2:
        st.subheader("Bounce Rate by Platform")
        bounce_by_platform = analyzer.bounce_rate_by_platform()
        if len(bounce_by_platform) > 0:
            st.dataframe(
                bounce_by_platform.style.format({
                    'sessions': '{:,}',
                    'bounce_rate': '{:.1f}%',
                    'conversion_rate': '{:.2f}%',
                    'avg_duration': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )

    # Sessions by hour
    st.subheader("Sessions by Hour")
    by_hour = analyzer.sessions_by_hour()
    if len(by_hour) > 0:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Bar(x=by_hour['hour'], y=by_hour['sessions'], name='Sessions'),
            secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=by_hour['hour'], y=by_hour['conversion_rate'], name='CVR %', mode='lines+markers'),
            secondary_y=True
        )
        fig.update_layout(height=350)
        fig.update_xaxes(title_text="Hour")
        fig.update_yaxes(title_text="Sessions", secondary_y=False)
        fig.update_yaxes(title_text="CVR %", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)


def show_funnel_deep_dive(df):
    """Show detailed funnel analysis."""
    from analytics import FunnelAnalyzer

    with st.spinner("Analyzing funnel..."):
        try:
            analyzer = FunnelAnalyzer(df)
        except Exception as e:
            st.error(f"Error analyzing funnel: {e}")
            return

    st.subheader("Conversion Funnel")

    # Main funnel
    funnel = analyzer.get_funnel()

    col1, col2 = st.columns([2, 1])

    with col1:
        # Funnel visualization
        fig = go.Figure(go.Funnel(
            y=funnel['step'],
            x=funnel['users'],
            textinfo="value+percent initial"
        ))
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("**Step-by-Step Metrics**")
        st.dataframe(
            funnel[['step', 'users', 'step_conversion', 'drop_off_rate']].style.format({
                'users': '{:,}',
                'step_conversion': '{:.1f}%',
                'drop_off_rate': '{:.1f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Drop-off analysis
    st.subheader("Drop-off Analysis: Where & Why Users Leave")

    drop_offs = analyzer.get_drop_off_analysis()
    if len(drop_offs) > 0:
        # Highlight biggest leaks
        fig = px.bar(
            drop_offs,
            x='from_step',
            y='drop_rate',
            color='drop_rate',
            color_continuous_scale='Reds',
            labels={'from_step': 'After Step', 'drop_rate': 'Drop Rate %'}
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Where do dropped users go instead?**")
        st.dataframe(
            drop_offs[['from_step', 'users_at_step', 'dropped', 'drop_rate', 'top_alternative_1', 'alt_1_count']].style.format({
                'users_at_step': '{:,}',
                'dropped': '{:,}',
                'drop_rate': '{:.1f}%',
                'alt_1_count': '{:,}'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    # Time between steps
    st.subheader("Time Between Funnel Steps")
    time_between = analyzer.get_time_between_steps()
    if len(time_between) > 0:
        col1, col2 = st.columns(2)
        with col1:
            st.dataframe(
                time_between.style.format({
                    'users': '{:,}',
                    'avg_minutes': '{:.1f}',
                    'median_minutes': '{:.1f}',
                    'p75_minutes': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        with col2:
            fig = px.bar(
                time_between,
                x='from_step',
                y='median_minutes',
                labels={'from_step': 'Step', 'median_minutes': 'Median Minutes'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Funnel by platform
    st.markdown("---")
    st.subheader("Funnel by Platform")

    funnel_by_platform = analyzer.get_funnel_by_segment('platform')
    if len(funnel_by_platform) > 0:
        fig = px.line(
            funnel_by_platform,
            x='step',
            y='pct_of_total',
            color='segment',
            markers=True,
            labels={'step': 'Funnel Step', 'pct_of_total': '% of Entry', 'segment': 'Platform'}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def show_path_analysis(df):
    """Show user journey path analysis."""
    from analytics import PathAnalyzer

    with st.spinner("Analyzing user paths..."):
        try:
            analyzer = PathAnalyzer(df)
        except Exception as e:
            st.error(f"Error analyzing paths: {e}")
            return

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Entry Points")
        entry_points = analyzer.get_entry_points()
        if len(entry_points) > 0:
            fig = px.bar(
                entry_points.head(10),
                x='users',
                y='entry_event',
                orientation='h',
                color='conversion_rate',
                color_continuous_scale='Greens',
                labels={'users': 'Users', 'entry_event': 'Entry Event', 'conversion_rate': 'CVR %'}
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Exit Points")
        exit_points = analyzer.get_exit_points()
        if len(exit_points) > 0:
            fig = px.bar(
                exit_points.head(10),
                x='users',
                y='exit_event',
                orientation='h',
                labels={'users': 'Users', 'exit_event': 'Exit Event'},
                color_discrete_sequence=['#FF6B6B']
            )
            fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Common paths
    st.subheader("Most Common User Paths")
    common_paths = analyzer.get_common_paths(n=15)
    if len(common_paths) > 0:
        st.dataframe(
            common_paths.style.format({
                'users': '{:,}',
                'pct': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Paths to Conversion")
        conversion_paths = analyzer.get_paths_to_conversion(n=10)
        if len(conversion_paths) > 0:
            st.dataframe(
                conversion_paths.style.format({
                    'users': '{:,}',
                    'pct_of_converters': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

    with col2:
        st.subheader("Event Flow Probabilities")
        event_flow = analyzer.get_event_flow()
        if len(event_flow) > 0:
            # Filter to key events
            key_events = ['homepage_viewed', 'product_page_viewed', 'product_added', 'cart_page_viewed']
            filtered_flow = event_flow[event_flow['event_type'].isin(key_events)]
            st.dataframe(
                filtered_flow.style.format({
                    'count': '{:,}',
                    'probability': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

    # Drop-off path analysis
    st.markdown("---")
    st.subheader("What Happens After Cart Abandonment?")
    st.markdown("Users who added to cart but didn't checkout - where did they go?")

    drop_paths = analyzer.get_paths_to_drop_off(drop_after='product_added', n=10)
    if len(drop_paths) > 0:
        st.dataframe(
            drop_paths.style.format({
                'users': '{:,}',
                'pct': '{:.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )


def show_merchant_intelligence(df):
    """Show merchant intelligence dashboard."""
    render_page_header("Merchant Intelligence", "Performance insights and health scores")

    from analytics import MerchantIntelligence

    with st.spinner("Analyzing merchants..."):
        try:
            intel = MerchantIntelligence(df)
        except Exception as e:
            st.error(f"Error analyzing merchants: {e}")
            return

    # Summary
    summary = intel.get_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Merchants", f"{summary['total_merchants']:,}")
    with col2:
        st.metric("Total Orders", f"{summary['total_orders']:,}")
    with col3:
        st.metric("Total Customers", f"{summary['total_customers']:,}")
    with col4:
        st.metric("Avg Repeat Rate", f"{summary['avg_repeat_rate']:.1f}%")
    with col5:
        st.metric("Top Category", summary['top_category'])

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Health Scores", "Category Analysis", "At-Risk Merchants", "Merchant Lookup"])

    with tab1:
        st.subheader("Merchant Health Scores")
        st.markdown("Composite score based on volume, retention, growth, and value metrics")

        health_scores = intel.get_merchant_health_scores(top_n=30)
        if len(health_scores) > 0:
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.bar(
                    health_scores.head(20),
                    x='health_score',
                    y='merchant_name',
                    orientation='h',
                    color='health_tier',
                    color_discrete_map={
                        'Thriving': '#2ECC71',
                        'Healthy': '#3498DB',
                        'Needs Attention': '#F39C12',
                        'At Risk': '#E74C3C'
                    },
                    labels={'health_score': 'Health Score', 'merchant_name': 'Merchant'}
                )
                fig.update_layout(height=600, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("**Health Tier Distribution**")
                tier_counts = health_scores['health_tier'].value_counts()
                fig = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    color=tier_counts.index,
                    color_discrete_map={
                        'Thriving': '#2ECC71',
                        'Healthy': '#3498DB',
                        'Needs Attention': '#F39C12',
                        'At Risk': '#E74C3C'
                    }
                )
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                health_scores.style.format({
                    'health_score': '{:.1f}',
                    'order_count': '{:,}',
                    'unique_customers': '{:,}',
                    'repeat_rate': '{:.1f}%',
                    'orders_per_customer': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Export Health Scores",
                health_scores.to_csv(index=False),
                "merchant_health_scores.csv",
                "text/csv"
            )

    with tab2:
        st.subheader("Category Performance")

        category_perf = intel.get_category_performance()
        if len(category_perf) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    category_perf.head(8),
                    values='total_orders',
                    names='category',
                    title='Market Share by Category',
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    category_perf,
                    x='category',
                    y='avg_repeat_rate',
                    color='avg_orders_per_customer',
                    color_continuous_scale='Blues',
                    labels={'category': 'Category', 'avg_repeat_rate': 'Avg Repeat Rate %'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                category_perf.style.format({
                    'merchant_count': '{:,}',
                    'total_orders': '{:,}',
                    'avg_orders_per_merchant': '{:.1f}',
                    'total_customers': '{:,}',
                    'total_revenue': 'QAR {:,.0f}',
                    'avg_repeat_rate': '{:.1f}%',
                    'avg_orders_per_customer': '{:.2f}',
                    'market_share': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")
        st.subheader("Top Merchants by Category")

        top_by_cat = intel.get_top_merchants_by_category(n_per_category=3)
        if len(top_by_cat) > 0:
            st.dataframe(
                top_by_cat.style.format({
                    'order_count': '{:,}',
                    'unique_customers': '{:,}',
                    'repeat_rate': '{:.1f}%',
                    'orders_per_customer': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

    with tab3:
        st.subheader("At-Risk Merchants")
        st.markdown("Merchants showing signs of decline or poor performance")

        at_risk = intel.get_at_risk_merchants(min_orders=10)
        if len(at_risk) > 0:
            st.warning(f"Found {len(at_risk)} merchants with risk indicators")

            st.dataframe(
                at_risk.style.format({
                    'risk_score': '{:.0f}',
                    'order_count': '{:,}',
                    'repeat_rate': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Export At-Risk Merchants",
                at_risk.to_csv(index=False),
                "at_risk_merchants.csv",
                "text/csv"
            )
        else:
            st.success("No at-risk merchants detected")

        st.markdown("---")
        st.subheader("New/Emerging Merchants")

        new_merchants = intel.get_new_merchants(days_threshold=7)
        if len(new_merchants) > 0:
            st.info(f"Found {len(new_merchants)} new merchants in the last 7 days")
            st.dataframe(
                new_merchants.style.format({
                    'order_count': '{:,}',
                    'unique_customers': '{:,}',
                    'orders_per_customer': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No new merchants in the selected period")

    with tab4:
        st.subheader("Merchant Lookup")

        merchants = intel.merchant_metrics['merchant_name'].sort_values().tolist()
        selected_merchant = st.selectbox("Select Merchant:", merchants)

        if selected_merchant:
            comparison = intel.get_merchant_comparison(selected_merchant)

            if 'error' not in comparison:
                st.markdown(f"### {selected_merchant}")
                st.markdown(f"**Category:** {comparison['category']}")

                col1, col2, col3, col4 = st.columns(4)

                metrics = comparison['metrics']

                with col1:
                    st.metric(
                        "Orders",
                        f"{metrics['order_count']['value']:,}",
                        f"Top {100-metrics['order_count']['overall_percentile']:.0f}%"
                    )
                with col2:
                    delta_color = "normal" if metrics['repeat_rate']['status'] == 'Above Average' else "inverse"
                    st.metric(
                        "Repeat Rate",
                        f"{metrics['repeat_rate']['value']:.1f}%",
                        metrics['repeat_rate']['status']
                    )
                with col3:
                    st.metric(
                        "Orders/Customer",
                        f"{metrics['orders_per_customer']['value']:.2f}",
                        f"Cat avg: {metrics['orders_per_customer']['category_avg']:.2f}"
                    )
                with col4:
                    st.metric(
                        "Avg Order Value",
                        f"QAR {metrics['avg_order_value']['value']:.0f}",
                        f"Cat avg: QAR {metrics['avg_order_value']['category_avg']:.0f}"
                    )

                st.markdown("---")
                st.subheader("Time Performance")
                time_perf = intel.get_time_performance(selected_merchant)
                if len(time_perf) > 0:
                    fig = px.bar(
                        time_perf,
                        x='time_slot',
                        y='orders',
                        color='pct_of_orders',
                        color_continuous_scale='Blues',
                        labels={'time_slot': 'Time Slot', 'orders': 'Orders'}
                    )
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)


def show_promo_analytics(df):
    """Show promo and marketing analytics."""
    render_page_header("Promo Analytics", "Promotional effectiveness and ROI")

    from analytics import PromoAnalyzer

    with st.spinner("Analyzing promo data..."):
        try:
            analyzer = PromoAnalyzer(df)
        except Exception as e:
            st.error(f"Error analyzing promos: {e}")
            return

    # Summary
    summary = analyzer.get_promo_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Orders", f"{summary['total_orders']:,}")
    with col2:
        st.metric("Promo Orders", f"{summary['promo_orders']:,}")
    with col3:
        st.metric("Promo Rate", f"{summary['promo_rate']:.1f}%")
    with col4:
        st.metric("Unique Promos", f"{summary['unique_promo_codes']:,}")
    with col5:
        if summary['total_discount_given'] > 0:
            st.metric("Total Discounts", f"QAR {summary['total_discount_given']:,.0f}")
        else:
            st.metric("Promo Users", f"{summary['promo_users']:,}")

    st.markdown("---")

    if summary['promo_orders'] == 0:
        st.info("No promo codes found in the current dataset. This could mean:")
        st.markdown("""
        - The data doesn't include promo code information
        - No promo codes were used during this period
        - Promo data is stored in a different format
        """)
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Promo Performance", "Acquisition & Retention", "Cannibalization", "By Merchant"])

    with tab1:
        st.subheader("Promo Code Performance")

        promo_perf = analyzer.get_promo_performance(min_uses=3)
        if 'message' not in promo_perf.columns and len(promo_perf) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    promo_perf.head(15),
                    x='uses',
                    y='promo_code',
                    orientation='h',
                    color='revenue_per_discount',
                    color_continuous_scale='Greens',
                    labels={'uses': 'Uses', 'promo_code': 'Promo Code', 'revenue_per_discount': 'Rev/Discount'}
                )
                fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.scatter(
                    promo_perf,
                    x='uses',
                    y='revenue_per_discount',
                    size='total_revenue',
                    hover_name='promo_code',
                    labels={'uses': 'Total Uses', 'revenue_per_discount': 'Revenue per Discount $'}
                )
                fig.update_layout(height=450)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                promo_perf.style.format({
                    'uses': '{:,}',
                    'unique_users': '{:,}',
                    'uses_per_user': '{:.2f}',
                    'total_revenue': 'QAR {:,.0f}',
                    'avg_order_value': 'QAR {:.0f}',
                    'total_discount': 'QAR {:,.0f}',
                    'avg_discount': 'QAR {:.0f}',
                    'discount_rate': '{:.1f}%',
                    'first_order_rate': '{:.1f}%',
                    'revenue_per_discount': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Export Promo Performance",
                promo_perf.to_csv(index=False),
                "promo_performance.csv",
                "text/csv"
            )
        else:
            st.info("Not enough promo data for detailed analysis")

        st.markdown("---")
        st.subheader("Promo vs Organic Orders")

        promo_organic = analyzer.get_promo_vs_organic()
        if len(promo_organic) > 0:
            st.dataframe(
                promo_organic.style.format({
                    'orders': '{:,}',
                    'unique_users': '{:,}',
                    'avg_order_value': 'QAR {:.0f}',
                    'unique_merchants': '{:,}',
                    'orders_per_user': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

    with tab2:
        st.subheader("First Order Acquisition Promos")
        st.markdown("Which promos are best at acquiring new customers who come back?")

        first_order_promos = analyzer.get_first_order_promos()
        if 'message' not in first_order_promos.columns and len(first_order_promos) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    first_order_promos.head(10),
                    x='first_orders',
                    y='promo_code',
                    orientation='h',
                    color='retention_rate',
                    color_continuous_scale='Greens',
                    labels={'first_orders': 'First Orders', 'promo_code': 'Promo', 'retention_rate': 'Retention %'}
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Scatter: acquisition vs retention
                fig = px.scatter(
                    first_order_promos,
                    x='first_orders',
                    y='retention_rate',
                    size='users_returned',
                    hover_name='promo_code',
                    labels={'first_orders': 'New Customers Acquired', 'retention_rate': 'Retention Rate %'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                first_order_promos.style.format({
                    'first_orders': '{:,}',
                    'avg_order_value': 'QAR {:.0f}',
                    'avg_discount': 'QAR {:.0f}',
                    'users_acquired': '{:,}',
                    'users_returned': '{:,}',
                    'retention_rate': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No first-order promo data available")

    with tab3:
        st.subheader("Promo Dependency Analysis")
        st.markdown("Are users becoming dependent on promos?")

        cannibalization = analyzer.get_promo_cannibalization()
        if len(cannibalization) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    cannibalization,
                    values='users',
                    names='user_type',
                    title='User Promo Behavior',
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.bar(
                    cannibalization,
                    x='user_type',
                    y='avg_orders',
                    color='avg_order_value',
                    color_continuous_scale='Blues',
                    labels={'user_type': 'User Type', 'avg_orders': 'Avg Orders', 'avg_order_value': 'AOV'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                cannibalization.style.format({
                    'users': '{:,}',
                    'avg_orders': '{:.1f}',
                    'avg_order_value': 'QAR {:.0f}',
                    'pct_of_users': '{:.1f}%'
                }),
                use_container_width=True,
                hide_index=True
            )

            # Recommendations
            st.markdown("---")
            st.subheader("Recommendations")

            recommendations = analyzer.get_recommendations()
            for rec in recommendations:
                if rec['type'] == 'warning':
                    st.warning(f"**{rec['title']}**: {rec['message']}")
                    st.markdown(f"*Action: {rec['action']}*")
                else:
                    st.info(f"**{rec['title']}**: {rec['message']}")
                    st.markdown(f"*Action: {rec['action']}*")

    with tab4:
        st.subheader("Promo Usage by Merchant")

        merchant_promo = analyzer.get_promo_by_merchant(top_n=20)
        if len(merchant_promo) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    merchant_promo.head(15),
                    x='promo_orders',
                    y='merchant_name',
                    orientation='h',
                    color='promo_rate',
                    color_continuous_scale='Oranges',
                    labels={'promo_orders': 'Promo Orders', 'merchant_name': 'Merchant', 'promo_rate': 'Promo Rate %'}
                )
                fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Merchants with highest promo rates
                high_promo = merchant_promo.nlargest(10, 'promo_rate')
                fig = px.bar(
                    high_promo,
                    x='promo_rate',
                    y='merchant_name',
                    orientation='h',
                    color='discount_pct_revenue',
                    color_continuous_scale='Reds',
                    labels={'promo_rate': 'Promo Rate %', 'merchant_name': 'Merchant'}
                )
                fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'}, title='Highest Promo Rate')
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                merchant_promo.style.format({
                    'total_orders': '{:,}',
                    'promo_orders': '{:,}',
                    'promo_rate': '{:.1f}%',
                    'total_discount': 'QAR {:,.0f}',
                    'total_revenue': 'QAR {:,.0f}',
                    'discount_pct_revenue': '{:.2f}%'
                }),
                use_container_width=True,
                hide_index=True
            )


def show_search_analytics(df):
    """Show search analytics dashboard."""
    render_page_header("Search Analytics", "Search behavior and conversion insights")

    from analytics import SearchAnalyzer

    with st.spinner("Analyzing search data..."):
        try:
            analyzer = SearchAnalyzer(df)
        except Exception as e:
            st.error(f"Error analyzing searches: {e}")
            return

    if not analyzer.has_data:
        st.warning("No search data found in the dataset.")
        st.info("Search analytics requires 'search_made' events with search query data.")
        return

    # Summary
    summary = analyzer.get_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Searches", f"{summary['total_searches']:,}")
    with col2:
        st.metric("Unique Searchers", f"{summary['unique_searchers']:,}")
    with col3:
        st.metric("Zero Result Rate", f"{summary['zero_result_rate']:.1f}%")
    with col4:
        st.metric("Search-to-Order", f"{summary['search_to_order_rate']:.1f}%")
    with col5:
        st.metric("Avg Results", f"{summary['avg_results_per_search']:.1f}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Top Searches", "Zero Results", "Search Funnel", "Recommendations"])

    with tab1:
        st.subheader("Most Popular Searches")

        top_searches = analyzer.get_top_searches(n=30)
        if len(top_searches) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    top_searches.head(15),
                    x='searches',
                    y='query',
                    orientation='h',
                    color='conversion_rate',
                    color_continuous_scale='Greens',
                    labels={'searches': 'Searches', 'query': 'Search Query', 'conversion_rate': 'CVR %'}
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    top_searches.style.format({
                        'searches': '{:,}',
                        'unique_users': '{:,}',
                        'avg_results': '{:.1f}',
                        'zero_result_rate': '{:.1f}%',
                        'conversion_rate': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            st.download_button(
                "📥 Export Top Searches",
                top_searches.to_csv(index=False),
                "top_searches.csv",
                "text/csv"
            )

        # Search by hour
        st.markdown("---")
        st.subheader("Search Activity by Hour")

        by_hour = analyzer.get_search_by_hour()
        if len(by_hour) > 0:
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(
                go.Bar(x=by_hour['hour'], y=by_hour['searches'], name='Searches'),
                secondary_y=False
            )
            fig.add_trace(
                go.Scatter(x=by_hour['hour'], y=by_hour['zero_result_rate'], name='Zero Result %', mode='lines+markers'),
                secondary_y=True
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Zero Result Searches - Optimization Opportunities")
        st.markdown("These searches returned no results. Consider adding these items or improving search matching.")

        zero_results = analyzer.get_zero_result_searches(n=30)
        if len(zero_results) > 0 and 'query' in zero_results.columns:
            col1, col2 = st.columns([2, 1])

            with col1:
                fig = px.bar(
                    zero_results.head(15),
                    x='occurrences',
                    y='query',
                    orientation='h',
                    color='priority',
                    color_discrete_map={'Critical': '#E74C3C', 'High': '#F39C12', 'Medium': '#3498DB', 'Low': '#95A5A6'},
                    labels={'occurrences': 'Times Searched', 'query': 'Query'}
                )
                fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    zero_results.style.format({
                        'occurrences': '{:,}',
                        'unique_users': '{:,}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            st.download_button(
                "📥 Export Zero Result Searches",
                zero_results.to_csv(index=False),
                "zero_result_searches.csv",
                "text/csv"
            )
        else:
            st.success("No zero-result searches found - great search coverage!")

    with tab3:
        st.subheader("Search to Order Funnel")

        funnel = analyzer.get_search_conversion_funnel()
        if len(funnel) > 0:
            fig = go.Figure(go.Funnel(
                y=funnel['step'],
                x=funnel['searchers'],
                textinfo="value+percent initial"
            ))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Categories
        st.markdown("---")
        st.subheader("Search Categories")

        categories = analyzer.get_search_categories()
        if len(categories) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    categories,
                    values='searches',
                    names='category',
                    title='Search Distribution by Category',
                    hole=0.4
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    categories.style.format({
                        'searches': '{:,}',
                        'unique_users': '{:,}',
                        'zero_result_rate': '{:.1f}%',
                        'pct_of_searches': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

    with tab4:
        st.subheader("Search Optimization Recommendations")

        recommendations = analyzer.get_search_recommendations()
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.warning(f"**{rec['title']}**\n\n{rec['message']}")
                st.markdown(f"*Recommended action: {rec['action']}*")
            elif rec['type'] == 'opportunity':
                st.info(f"**{rec['title']}**\n\n{rec['message']}")
                st.markdown(f"*Recommended action: {rec['action']}*")
            else:
                st.success(f"**{rec['title']}**\n\n{rec['message']}")
            st.markdown("---")


def show_delivery_analytics(df):
    """Show delivery and fulfillment analytics."""
    render_page_header("Delivery Analytics", "Fulfillment performance and timing")

    from analytics import DeliveryAnalyzer

    with st.spinner("Analyzing delivery data..."):
        try:
            analyzer = DeliveryAnalyzer(df)
        except Exception as e:
            st.error(f"Error analyzing delivery: {e}")
            return

    if not analyzer.has_data:
        st.warning("No order data found.")
        return

    # Summary
    summary = analyzer.get_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Orders", f"{summary['total_orders']:,}")
    with col2:
        st.metric("Unique Customers", f"{summary['unique_customers']:,}")
    with col3:
        if summary['has_delivery_data']:
            st.metric("Avg Delivery Time", f"{summary['avg_delivery_time_min']:.0f} min")
        else:
            st.metric("Has Delivery Data", "No")
    with col4:
        if summary['has_delivery_data']:
            st.metric("Median Delivery", f"{summary['median_delivery_time_min']:.0f} min")
        else:
            st.metric("Order Types", len(summary.get('order_types', {})))
    with col5:
        if summary['has_delivery_data']:
            st.metric("Fulfillment Rate", f"{summary['fulfillment_rate']:.1f}%")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Delivery Times", "Peak Hours", "By Merchant", "Recommendations"])

    with tab1:
        if analyzer.has_delivery_events:
            st.subheader("Delivery Time Distribution")

            dist = analyzer.get_delivery_time_distribution()
            if len(dist) > 0 and 'delivery_time' in dist.columns:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        dist,
                        x='delivery_time',
                        y='orders',
                        color='pct',
                        color_continuous_scale='Blues',
                        labels={'delivery_time': 'Delivery Time', 'orders': 'Orders'}
                    )
                    fig.update_layout(height=350)
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.dataframe(
                        dist.style.format({
                            'orders': '{:,}',
                            'pct': '{:.1f}%'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )

            # Delivery impact on reorders
            st.markdown("---")
            st.subheader("Delivery Time Impact on Reorders")

            impact = analyzer.get_delivery_impact_on_reorders()
            if len(impact) > 0 and 'delivery_time' in impact.columns:
                fig = px.bar(
                    impact,
                    x='delivery_time',
                    y='reorder_rate',
                    color='reorder_rate',
                    color_continuous_scale='Greens',
                    labels={'delivery_time': 'Delivery Time', 'reorder_rate': 'Reorder Rate %'}
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    impact.style.format({
                        'orders': '{:,}',
                        'unique_users': '{:,}',
                        'reorder_rate': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("No delivery event data available. Add 'order_delivered' events for delivery time analysis.")

        # Order types
        st.markdown("---")
        st.subheader("Order Type Analysis")

        order_types = analyzer.get_order_type_analysis()
        if len(order_types) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.pie(
                    order_types,
                    values='orders',
                    names='order_type',
                    title='Orders by Type',
                    hole=0.4
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    order_types.style.format({
                        'orders': '{:,}',
                        'unique_customers': '{:,}',
                        'avg_order_value': 'QAR {:.0f}',
                        'avg_delivery_fee': 'QAR {:.1f}',
                        'pct_of_orders': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

    with tab2:
        st.subheader("Peak Order Hours")

        peak_hours = analyzer.get_peak_hours()
        if len(peak_hours) > 0:
            fig = px.bar(
                peak_hours,
                x='hour',
                y='avg_orders',
                color='is_peak',
                color_discrete_map={True: '#E74C3C', False: '#3498DB'},
                labels={'hour': 'Hour', 'avg_orders': 'Avg Orders', 'is_peak': 'Peak Hour'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            peak_count = peak_hours['is_peak'].sum()
            st.info(f"Identified {peak_count} peak hours requiring higher capacity")

            st.dataframe(
                peak_hours.style.format({
                    'avg_orders': '{:.1f}',
                    'median_orders': '{:.0f}',
                    'max_orders': '{:.0f}',
                    'p95_orders': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )

        # Delivery by hour
        st.markdown("---")
        st.subheader("Orders by Hour")

        by_hour = analyzer.get_delivery_by_hour()
        if len(by_hour) > 0:
            fig = px.bar(
                by_hour,
                x='hour',
                y='orders',
                color='time_slot',
                labels={'hour': 'Hour', 'orders': 'Orders', 'time_slot': 'Time Slot'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Merchant Delivery Performance")

        merchant_perf = analyzer.get_merchant_delivery_performance(min_orders=10)
        if len(merchant_perf) > 0:
            if 'avg_delivery_time' in merchant_perf.columns:
                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        merchant_perf.nsmallest(15, 'avg_delivery_time'),
                        x='avg_delivery_time',
                        y='merchant_name',
                        orientation='h',
                        color='avg_delivery_time',
                        color_continuous_scale='Greens_r',
                        labels={'avg_delivery_time': 'Avg Delivery (min)', 'merchant_name': 'Merchant'},
                        title='Fastest Merchants'
                    )
                    fig.update_layout(height=450, yaxis={'categoryorder': 'total descending'})
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        merchant_perf.nlargest(15, 'avg_delivery_time'),
                        x='avg_delivery_time',
                        y='merchant_name',
                        orientation='h',
                        color='avg_delivery_time',
                        color_continuous_scale='Reds',
                        labels={'avg_delivery_time': 'Avg Delivery (min)', 'merchant_name': 'Merchant'},
                        title='Slowest Merchants'
                    )
                    fig.update_layout(height=450, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                merchant_perf.head(30),
                use_container_width=True,
                hide_index=True
            )

    with tab4:
        st.subheader("Delivery Recommendations")

        recommendations = analyzer.get_recommendations()
        for rec in recommendations:
            if rec['type'] == 'warning':
                st.warning(f"**{rec['title']}**\n\n{rec['message']}")
            elif rec['type'] == 'insight':
                st.success(f"**{rec['title']}**\n\n{rec['message']}")
            else:
                st.info(f"**{rec['title']}**\n\n{rec['message']}")
            st.markdown(f"*Action: {rec['action']}*")
            st.markdown("---")


def show_customer_scoring(df):
    """Show customer scoring and journey analytics."""
    render_page_header("Customer Scoring", "Multi-dimensional customer health scores")

    from analytics import CustomerScorer

    with st.spinner("Scoring customers..."):
        try:
            scorer = CustomerScorer(df)
        except Exception as e:
            st.error(f"Error scoring customers: {e}")
            return

    # Summary
    summary = scorer.get_summary()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Customers", f"{summary['total_customers']:,}")
    with col2:
        st.metric("With Orders", f"{summary['with_orders']:,}")
    with col3:
        st.metric("Avg Health Score", f"{summary['avg_health_score']:.1f}")
    with col4:
        st.metric("Champions", f"{summary['champions']:,}")
    with col5:
        st.metric("At Risk", f"{summary['at_risk']:,}")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Health Scores", "Lifecycle Stages", "Next Best Action", "Export"])

    with tab1:
        st.subheader("Customer Health Scores")

        health = scorer.calculate_health_score()

        col1, col2 = st.columns(2)

        with col1:
            # Health tier distribution
            tier_counts = health['health_tier'].value_counts()
            fig = px.pie(
                values=tier_counts.values,
                names=tier_counts.index,
                title='Health Tier Distribution',
                color=tier_counts.index,
                color_discrete_map={
                    'Champion': '#2ECC71',
                    'Healthy': '#3498DB',
                    'At Risk': '#F39C12',
                    'Critical': '#E74C3C'
                },
                hole=0.4
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Health score distribution
            fig = px.histogram(
                health,
                x='health_score',
                nbins=20,
                color='health_tier',
                color_discrete_map={
                    'Champion': '#2ECC71',
                    'Healthy': '#3498DB',
                    'At Risk': '#F39C12',
                    'Critical': '#E74C3C'
                },
                labels={'health_score': 'Health Score', 'count': 'Customers'}
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # High value at risk
        st.markdown("---")
        st.subheader("High-Value Customers at Risk")

        at_risk = scorer.get_high_value_at_risk(n=50)
        if len(at_risk) > 0:
            st.warning(f"Found {len(at_risk)} repeat customers at risk of churning")
            st.dataframe(
                at_risk.style.format({
                    'health_score': '{:.1f}',
                    'order_count': '{:,}',
                    'days_since_last': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Export At-Risk Customers",
                at_risk.to_csv(index=False),
                "at_risk_customers.csv",
                "text/csv"
            )

    with tab2:
        st.subheader("Customer Lifecycle Stages")

        segment_summary = scorer.get_segment_summary()
        if len(segment_summary) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    segment_summary,
                    x='users',
                    y='lifecycle_stage',
                    orientation='h',
                    color='avg_health_score',
                    color_continuous_scale='RdYlGn',
                    labels={'users': 'Users', 'lifecycle_stage': 'Stage', 'avg_health_score': 'Avg Health'}
                )
                fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(
                    segment_summary.style.format({
                        'users': '{:,}',
                        'avg_health_score': '{:.1f}',
                        'avg_orders': '{:.1f}',
                        'avg_days_inactive': '{:.0f}',
                        'pct_of_users': '{:.1f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

        # Conversion opportunities
        st.markdown("---")
        st.subheader("Conversion Opportunities")
        st.markdown("Highly engaged browsers who haven't ordered yet")

        opportunities = scorer.get_conversion_opportunities(n=50)
        if len(opportunities) > 0:
            st.success(f"Found {len(opportunities)} high-potential conversion opportunities")
            st.dataframe(
                opportunities.style.format({
                    'engagement_score': '{:.1f}',
                    'depth_score': '{:.1f}',
                    'recency_score': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Export Conversion Opportunities",
                opportunities.to_csv(index=False),
                "conversion_opportunities.csv",
                "text/csv"
            )

    with tab3:
        st.subheader("Next Best Action Recommendations")

        nba = scorer.get_next_best_action()
        if len(nba) > 0:
            # Summary by action
            action_summary = nba.groupby(['action', 'priority']).size().reset_index(name='users')
            action_summary = action_summary.sort_values('users', ascending=False)

            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    action_summary,
                    x='users',
                    y='action',
                    orientation='h',
                    color='priority',
                    color_discrete_map={'High': '#E74C3C', 'Medium': '#F39C12', 'Low': '#3498DB'},
                    labels={'users': 'Users', 'action': 'Recommended Action'}
                )
                fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.dataframe(action_summary, use_container_width=True, hide_index=True)

            # Sample of recommendations
            st.markdown("---")
            st.subheader("Sample Recommendations")

            sample = nba.head(100)
            st.dataframe(
                sample[['amplitude_id', 'lifecycle_stage', 'health_tier', 'action', 'channel', 'priority']],
                use_container_width=True,
                hide_index=True
            )

    with tab4:
        st.subheader("Export Customer Data for CRM")

        export = scorer.export_for_crm()

        st.info(f"Ready to export {len(export):,} customer records")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", f"{len(export):,}")
        with col2:
            st.metric("With Actions", f"{(export['action'].notna()).sum():,}")
        with col3:
            st.metric("High Priority", f"{(export['priority'] == 'High').sum():,}")

        st.dataframe(export.head(50), use_container_width=True, hide_index=True)

        st.download_button(
            "📥 Export Full CRM Data",
            export.to_csv(index=False),
            "customer_crm_export.csv",
            "text/csv"
        )


def show_anomaly_detection(df):
    """Show anomaly detection dashboard."""
    render_page_header("Anomaly Detection", "Unusual patterns and alerts")

    from analytics import AnomalyDetector

    # Sensitivity selector
    sensitivity = st.sidebar.slider("Detection Sensitivity", 1.5, 3.0, 2.0, 0.1,
                                    help="Lower = more sensitive (more anomalies)")

    with st.spinner("Detecting anomalies..."):
        try:
            detector = AnomalyDetector(df, sensitivity=sensitivity)
        except Exception as e:
            st.error(f"Error detecting anomalies: {e}")
            return

    # Summary
    summary = detector.get_anomaly_summary()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Volume Anomalies", summary['volume_anomalies'])
    with col2:
        st.metric("Suspicious Users", summary['suspicious_users'])
    with col3:
        st.metric("Merchant Anomalies", summary['merchant_anomalies'])
    with col4:
        st.metric("CVR Anomalies", summary['conversion_anomalies'])

    # Alerts
    st.markdown("---")
    st.subheader("Active Alerts")

    alerts = detector.get_alerts(severity_threshold='medium')
    if alerts:
        for alert in alerts:
            if alert['severity'] == 'high':
                st.error(f"🔴 **{alert['title']}**\n\n{alert['message']}\n\n*Action: {alert['action']}*")
            elif alert['severity'] == 'medium':
                st.warning(f"🟡 **{alert['title']}**\n\n{alert['message']}\n\n*Action: {alert['action']}*")
            else:
                st.info(f"🔵 **{alert['title']}**\n\n{alert['message']}\n\n*Action: {alert['action']}*")
    else:
        st.success("No significant anomalies detected")

    st.markdown("---")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Volume", "User Behavior", "Merchants", "Conversion"])

    with tab1:
        st.subheader("Volume Anomalies")

        volume = detector.detect_volume_anomalies()
        if len(volume) > 0 and 'datetime' in volume.columns:
            fig = px.scatter(
                volume,
                x='datetime',
                y='events',
                color='anomaly_type',
                size='z_score',
                color_discrete_map={'spike': '#E74C3C', 'drop': '#3498DB'},
                labels={'datetime': 'Time', 'events': 'Events', 'anomaly_type': 'Type'}
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                volume.style.format({
                    'events': '{:,}',
                    'z_score': '{:.2f}',
                    'expected_min': '{:.0f}',
                    'expected_max': '{:.0f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No volume anomalies detected")

    with tab2:
        st.subheader("Suspicious User Behavior")

        user_anomalies = detector.detect_user_behavior_anomalies()
        if len(user_anomalies) > 0 and 'amplitude_id' in user_anomalies.columns:
            st.warning(f"Found {len(user_anomalies)} users with suspicious behavior patterns")

            st.dataframe(
                user_anomalies.style.format({
                    'total_events': '{:,}',
                    'checkouts': '{:,}',
                    'events_per_hour': '{:.1f}',
                    'session_duration_hours': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )

            st.download_button(
                "📥 Export Suspicious Users",
                user_anomalies.to_csv(index=False),
                "suspicious_users.csv",
                "text/csv"
            )
        else:
            st.success("No suspicious user behavior detected")

    with tab3:
        st.subheader("Merchant Performance Anomalies")

        merchant_anomalies = detector.detect_merchant_anomalies()
        if len(merchant_anomalies) > 0 and 'merchant_name' in merchant_anomalies.columns:
            col1, col2 = st.columns(2)

            surges = merchant_anomalies[merchant_anomalies['anomaly_type'] == 'surge']
            drops = merchant_anomalies[merchant_anomalies['anomaly_type'] == 'drop']

            with col1:
                st.markdown("**Order Surges**")
                if len(surges) > 0:
                    st.dataframe(
                        surges.style.format({
                            'orders': '{:,}',
                            'avg_daily_orders': '{:.1f}',
                            'z_score': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No surges detected")

            with col2:
                st.markdown("**Order Drops**")
                if len(drops) > 0:
                    st.dataframe(
                        drops.style.format({
                            'orders': '{:,}',
                            'avg_daily_orders': '{:.1f}',
                            'z_score': '{:.2f}'
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No drops detected")
        else:
            st.success("No merchant anomalies detected")

    with tab4:
        st.subheader("Conversion Rate Anomalies")

        cvr_anomalies = detector.detect_conversion_anomalies()
        if len(cvr_anomalies) > 0:
            fig = px.scatter(
                cvr_anomalies,
                x='datetime',
                y='conversion_rate',
                color='anomaly_type',
                size='z_score',
                color_discrete_map={'high': '#2ECC71', 'low': '#E74C3C'},
                labels={'datetime': 'Time', 'conversion_rate': 'CVR %', 'anomaly_type': 'Type'}
            )
            fig.add_hline(y=cvr_anomalies['expected_cvr'].iloc[0], line_dash="dash", line_color="gray")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(
                cvr_anomalies.style.format({
                    'unique_users': '{:,}',
                    'checkout_users': '{:,}',
                    'conversion_rate': '{:.2f}%',
                    'expected_cvr': '{:.2f}%',
                    'z_score': '{:.2f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("No conversion anomalies detected")


def show_orders(df):
    """Show order analysis."""
    render_page_header("Order Analysis", "Order metrics and distributions")

    checkout_df = df[df['event_type'] == 'checkout_completed'].copy()
    checkout_df = parse_props(checkout_df)

    # Extract order properties
    checkout_df['order_total'] = checkout_df['props'].apply(
        lambda x: float(x.get('order_total', 0)) if x.get('order_total') else 0
    )
    checkout_df['order_type'] = checkout_df['props'].apply(
        lambda x: x.get('order_type', 'unknown')
    )
    checkout_df['merchant_name'] = checkout_df['props'].apply(
        lambda x: x.get('merchant_name', 'unknown')
    )

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Orders", f"{len(checkout_df):,}")
    with col2:
        st.metric("Total Revenue", f"QAR {checkout_df['order_total'].sum():,.0f}")
    with col3:
        st.metric("Avg Order Value", f"QAR {checkout_df['order_total'].mean():.0f}")
    with col4:
        st.metric("Unique Buyers", f"{checkout_df['amplitude_id'].nunique():,}")

    st.markdown("---")

    # Order type breakdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Order Types")
        order_types = checkout_df['order_type'].value_counts()
        fig = px.pie(values=order_types.values, names=order_types.index, hole=0.4)
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top Merchants")
        top_merchants = checkout_df['merchant_name'].value_counts().head(10)
        fig = px.bar(
            x=top_merchants.values,
            y=top_merchants.index,
            orientation='h',
            labels={'x': 'Orders', 'y': 'Merchant'}
        )
        fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)

    # Order value distribution
    st.subheader("Order Value Distribution")
    fig = px.histogram(
        checkout_df[checkout_df['order_total'] > 0],
        x='order_total',
        nbins=50,
        labels={'order_total': 'Order Value (QAR)'}
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)

    # Top orderers
    st.subheader("Top Orderers")
    top_orderers = checkout_df.groupby('amplitude_id').size().sort_values(ascending=False).head(10)
    top_orderers_df = pd.DataFrame({
        'User ID': top_orderers.index,
        'Orders': top_orderers.values
    })
    st.dataframe(top_orderers_df, use_container_width=True)


def show_platform(df):
    """Show platform analysis."""
    render_page_header("Platform Analysis", "iOS vs Android comparison")

    # Normalize platform
    df = df.copy()
    df['platform'] = df['platform'].str.lower()

    # Platform metrics
    platforms = df['platform'].unique()

    platform_data = []
    for platform in platforms:
        platform_df = df[df['platform'] == platform]
        homepage = platform_df[platform_df['event_type'] == 'homepage_viewed']['amplitude_id'].nunique()
        checkout = platform_df[platform_df['event_type'] == 'checkout_completed']['amplitude_id'].nunique()
        orders = len(platform_df[platform_df['event_type'] == 'checkout_completed'])
        cvr = checkout / homepage * 100 if homepage > 0 else 0
        opu = orders / checkout if checkout > 0 else 0

        platform_data.append({
            'Platform': platform.upper(),
            'Users': homepage,
            'Buyers': checkout,
            'Orders': orders,
            'CVR': cvr,
            'OPU': opu
        })

    platform_df = pd.DataFrame(platform_data)

    # Metrics table
    st.subheader("Platform Comparison")
    st.dataframe(platform_df.style.format({
        'Users': '{:,}',
        'Buyers': '{:,}',
        'Orders': '{:,}',
        'CVR': '{:.1f}%',
        'OPU': '{:.2f}'
    }), use_container_width=True)

    # Charts
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Users by Platform")
        fig = px.bar(platform_df, x='Platform', y='Users')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("CVR by Platform")
        fig = px.bar(platform_df, x='Platform', y='CVR')
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)


def show_hourly(df):
    """Show hourly trends."""
    render_page_header("Hourly Trends", "Activity patterns by time of day")

    df = df.copy()
    df['hour'] = pd.to_datetime(df['event_time']).dt.hour

    # Hourly homepage views
    homepage_hourly = df[df['event_type'] == 'homepage_viewed'].groupby('hour').size()
    checkout_hourly = df[df['event_type'] == 'checkout_completed'].groupby('hour').size()

    # Create hourly dataframe
    hourly_df = pd.DataFrame({
        'Hour': range(24),
        'Homepage Views': [homepage_hourly.get(h, 0) for h in range(24)],
        'Orders': [checkout_hourly.get(h, 0) for h in range(24)]
    })

    # Calculate CVR
    hourly_df['CVR'] = (hourly_df['Orders'] / hourly_df['Homepage Views'] * 100).fillna(0)

    # Traffic chart
    st.subheader("Traffic & Orders by Hour")
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(x=hourly_df['Hour'], y=hourly_df['Homepage Views'], name='Homepage Views'),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=hourly_df['Hour'], y=hourly_df['Orders'], name='Orders', mode='lines+markers'),
        secondary_y=True
    )

    fig.update_layout(height=400)
    fig.update_xaxes(title_text="Hour of Day")
    fig.update_yaxes(title_text="Homepage Views", secondary_y=False)
    fig.update_yaxes(title_text="Orders", secondary_y=True)

    st.plotly_chart(fig, use_container_width=True)

    # CVR by hour
    st.subheader("Conversion Rate by Hour")
    fig = px.line(hourly_df, x='Hour', y='CVR', markers=True)
    fig.update_layout(height=300)
    fig.update_yaxes(title_text="CVR (%)")
    st.plotly_chart(fig, use_container_width=True)

    # Peak hours
    col1, col2, col3 = st.columns(3)
    with col1:
        peak_traffic = hourly_df.loc[hourly_df['Homepage Views'].idxmax()]
        st.metric("Peak Traffic Hour", f"{int(peak_traffic['Hour']):02d}:00")
    with col2:
        peak_orders = hourly_df.loc[hourly_df['Orders'].idxmax()]
        st.metric("Peak Orders Hour", f"{int(peak_orders['Hour']):02d}:00")
    with col3:
        peak_cvr = hourly_df.loc[hourly_df['CVR'].idxmax()]
        st.metric("Peak CVR Hour", f"{int(peak_cvr['Hour']):02d}:00")


def show_attribution_modeling(df):
    """Show attribution modeling analysis."""
    render_page_header("Attribution Modeling", "Multi-touch conversion attribution")

    from analytics import AttributionModeler, ChannelAttributor

    try:
        modeler = AttributionModeler(df)
        channel_attr = ChannelAttributor(df)
    except Exception as e:
        st.error(f"Error initializing attribution modeler: {e}")
        return

    # Journey stats
    st.subheader("User Journey Statistics")
    stats = modeler.get_journey_stats()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Conversions", f"{stats['total_conversions']:,}")
    with col2:
        st.metric("Avg Journey Length", f"{stats['avg_journey_length']:.1f} steps")
    with col3:
        st.metric("Median Journey Length", f"{stats.get('median_journey_length', 0):.0f} steps")
    with col4:
        st.metric("Avg Duration", f"{stats['avg_journey_duration_mins']:.1f} mins")

    st.markdown("---")

    # Attribution model comparison
    st.subheader("Attribution Model Comparison")

    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Channel Influence", "Platform Attribution"])

    with tab1:
        comparison = modeler.compare_models()

        if len(comparison) > 0:
            # Melt for visualization
            comparison_melted = comparison.melt(
                id_vars=['touchpoint'],
                var_name='model',
                value_name='attribution_pct'
            )
            comparison_melted['model'] = comparison_melted['model'].str.replace('_pct', '')

            fig = px.bar(
                comparison_melted,
                x='touchpoint',
                y='attribution_pct',
                color='model',
                barmode='group',
                labels={'touchpoint': 'Touchpoint', 'attribution_pct': 'Attribution %', 'model': 'Model'},
                title='Attribution by Model'
            )
            fig.update_layout(height=500, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            # Table view
            st.dataframe(
                comparison.style.format({col: '{:.1f}%' for col in comparison.columns if col != 'touchpoint'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning("No journey data available for attribution analysis")

    with tab2:
        influence = modeler.get_channel_influence()

        if len(influence) > 0:
            col1, col2 = st.columns(2)

            with col1:
                fig = px.bar(
                    influence,
                    x='touchpoint',
                    y='total_appearances',
                    labels={'touchpoint': 'Touchpoint', 'total_appearances': 'Appearances in Journeys'},
                    title='Touchpoint Frequency'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Position analysis
                position_data = influence[['touchpoint', 'as_first_touch', 'as_last_touch', 'as_middle_touch']].melt(
                    id_vars=['touchpoint'],
                    var_name='position',
                    value_name='count'
                )
                fig = px.bar(
                    position_data,
                    x='touchpoint',
                    y='count',
                    color='position',
                    barmode='stack',
                    labels={'touchpoint': 'Touchpoint', 'count': 'Count', 'position': 'Position'},
                    title='Touchpoint Position in Journey'
                )
                fig.update_layout(height=400, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(influence, use_container_width=True, hide_index=True)
        else:
            st.warning("No channel influence data available")

    with tab3:
        platform_attr = channel_attr.platform_attribution()
        device_attr = channel_attr.device_attribution()

        if len(platform_attr) > 0:
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Platform Attribution**")
                fig = px.pie(
                    platform_attr,
                    values='conversions',
                    names='platform',
                    title='Conversions by Platform'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    platform_attr.style.format({
                        'conversions': '{:,}',
                        'conversion_pct': '{:.1f}%',
                        'total_users': '{:,}',
                        'cvr': '{:.2f}%'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

            with col2:
                if len(device_attr) > 0:
                    st.markdown("**Device Attribution**")
                    fig = px.bar(
                        device_attr,
                        x='conversions',
                        y='device_family',
                        orientation='h',
                        labels={'conversions': 'Conversions', 'device_family': 'Device'},
                        title='Conversions by Device'
                    )
                    fig.update_layout(height=350, yaxis={'categoryorder': 'total ascending'})
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No platform attribution data available")

    # Most common paths
    st.markdown("---")
    st.subheader("Most Common Conversion Paths")

    if stats['most_common_paths']:
        paths_df = pd.DataFrame(stats['most_common_paths'])
        st.dataframe(paths_df, use_container_width=True, hide_index=True)
    else:
        st.info("No common paths found")


def show_reactivation_targeting(df):
    """Show reactivation targeting analysis."""
    render_page_header("Reactivation Targeting", "Dormant user identification and campaigns")

    from analytics import ReactivationTargeter

    # Dormancy threshold selector
    dormancy_days = st.slider("Dormancy Threshold (days)", 7, 60, 14)

    try:
        targeter = ReactivationTargeter(df, dormancy_days=dormancy_days)
    except Exception as e:
        st.error(f"Error initializing reactivation targeter: {e}")
        return

    # Summary
    summary = targeter.get_reactivation_summary()

    st.subheader("Dormant User Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Dormant Users", f"{summary['total_dormant_users']:,}")
    with col2:
        st.metric("Avg Days Inactive", f"{summary.get('avg_days_inactive', 0):.0f}")
    with col3:
        st.metric("High Potential Users", f"{summary.get('high_potential_users', 0):,}")
    with col4:
        st.metric("Potential Revenue", f"QAR {summary.get('potential_recoverable_revenue', 0):,.0f}")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["Campaign Segments", "User Scoring", "Export"])

    with tab1:
        st.subheader("Recommended Campaign Segments")

        segments = targeter.get_reactivation_campaign_segments()

        if len(segments) > 0:
            for _, segment in segments.iterrows():
                with st.expander(f"**{segment['segment'].replace('_', ' ').title()}** ({segment['user_count']:,} users)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Avg Revenue:** QAR {segment['avg_revenue']:.0f}")
                    with col2:
                        st.markdown(f"**Avg Orders:** {segment['avg_orders']:.1f}")
                    with col3:
                        st.markdown(f"**Priority:** {segment['priority']}")

                    st.markdown(f"**Recommended Channel:** {segment['recommended_channel']}")
                    st.markdown(f"**Recommended Incentive:** {segment['recommended_incentive']}")
        else:
            st.info("No campaign segments found for the current dormancy threshold")

        # Dormancy distribution
        if summary['dormancy_distribution']:
            st.markdown("### Dormancy Distribution")
            dist_df = pd.DataFrame([
                {'Dormancy Tier': k, 'Users': v}
                for k, v in summary['dormancy_distribution'].items()
            ])
            fig = px.bar(dist_df, x='Dormancy Tier', y='Users', title='Users by Dormancy Duration')
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("User Reactivation Scores")

        scored = targeter.score_reactivation_potential()

        if len(scored) > 0:
            # Tier distribution
            col1, col2 = st.columns(2)

            with col1:
                tier_counts = scored['reactivation_tier'].value_counts()
                fig = px.pie(
                    values=tier_counts.values,
                    names=tier_counts.index,
                    title='Users by Reactivation Potential',
                    color_discrete_sequence=px.colors.sequential.RdBu
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = px.histogram(
                    scored,
                    x='reactivation_score',
                    nbins=20,
                    labels={'reactivation_score': 'Reactivation Score'},
                    title='Score Distribution'
                )
                fig.update_layout(height=350)
                st.plotly_chart(fig, use_container_width=True)

            # Top users table
            st.markdown("### Top Reactivation Targets")
            display_cols = ['user_id', 'days_inactive', 'total_orders', 'total_revenue',
                           'reactivation_score', 'reactivation_tier']
            available_cols = [c for c in display_cols if c in scored.columns]

            st.dataframe(
                scored[available_cols].head(50).style.format({
                    'total_revenue': 'QAR {:.0f}',
                    'reactivation_score': '{:.1f}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No scored users available")

    with tab3:
        st.subheader("Export for Campaign")

        segment_options = [
            "All Dormant Users",
            "high_value_recently_dormant",
            "frequent_orderers_churned",
            "one_time_buyers"
        ]

        selected_segment = st.selectbox("Select Segment:", segment_options)
        limit = st.number_input("Max users to export", min_value=100, max_value=10000, value=1000)

        if st.button("Generate Export"):
            segment_filter = None if selected_segment == "All Dormant Users" else selected_segment
            export_df = targeter.export_for_campaign(segment=segment_filter, limit=limit)

            if len(export_df) > 0:
                st.success(f"Generated export with {len(export_df):,} users")
                st.dataframe(export_df.head(100), use_container_width=True, hide_index=True)

                st.download_button(
                    "📥 Download Full Export (CSV)",
                    export_df.to_csv(index=False),
                    f"reactivation_{selected_segment}.csv",
                    "text/csv"
                )
            else:
                st.warning("No users found for the selected segment")


def show_product_affinity(df):
    """Show product affinity and bundle analysis."""
    render_page_header("Product Affinity", "Bundle recommendations and cross-sell")

    from analytics import ProductAffinityAnalyzer, MerchantCrossSeller

    try:
        affinity = ProductAffinityAnalyzer(df)
        cross_seller = MerchantCrossSeller(df)
    except Exception as e:
        st.error(f"Error initializing product affinity analyzer: {e}")
        return

    # Summary
    summary = affinity.get_affinity_summary()

    st.subheader("Transaction Summary")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Transactions", f"{summary['total_transactions']:,}")
    with col2:
        st.metric("Unique Products", f"{summary['unique_products']:,}")
    with col3:
        st.metric("Avg Basket Size", f"{summary['avg_basket_size']:.1f}")
    with col4:
        st.metric("Multi-item Rate", f"{summary.get('multi_item_order_rate', 0):.1f}%")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["Bundle Recommendations", "Association Rules", "Merchant Affinity", "Cross-sell"])

    with tab1:
        st.subheader("Top Bundle Recommendations")

        bundles = summary.get('top_bundles', [])

        if bundles:
            for bundle in bundles:
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.markdown(f"**{bundle['product_1']}** + **{bundle['product_2']}**")
                with col2:
                    st.metric("Lift", f"{bundle['lift']:.2f}")
                with col3:
                    st.metric("Confidence", f"{bundle['confidence']:.0f}%")
                st.caption(bundle['recommendation'])
                st.markdown("---")
        else:
            st.info("Not enough transaction data for bundle recommendations. Product details may not be available in the event data.")

        # Merchant bundles
        st.subheader("Top Bundles by Merchant")
        merchant_bundles = affinity.get_merchant_top_bundles()

        if len(merchant_bundles) > 0:
            st.dataframe(
                merchant_bundles.head(20).style.format({
                    'co_occurrence': '{:,}',
                    'total_orders': '{:,}'
                }),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No merchant bundle data available")

    with tab2:
        st.subheader("Association Rules")
        st.caption("Rules show: If customer buys X, they often also buy Y")

        min_support = st.slider("Minimum Support (%)", 0.1, 5.0, 1.0, 0.1) / 100
        min_confidence = st.slider("Minimum Confidence (%)", 5, 50, 10) / 100

        rules = affinity.calculate_association_rules(min_support=min_support, min_confidence=min_confidence)

        if len(rules) > 0:
            st.markdown(f"Found **{len(rules)}** association rules")

            # Filter by lift
            strong_rules = rules[rules['lift'] > 1].sort_values('lift', ascending=False)

            if len(strong_rules) > 0:
                st.dataframe(
                    strong_rules.head(50).style.format({
                        'support': '{:.2f}%',
                        'confidence': '{:.1f}%',
                        'lift': '{:.2f}',
                        'count': '{:,}'
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                # Visualization
                if len(strong_rules) >= 5:
                    fig = px.scatter(
                        strong_rules.head(50),
                        x='confidence',
                        y='lift',
                        size='count',
                        hover_data=['antecedent', 'consequent'],
                        labels={'confidence': 'Confidence %', 'lift': 'Lift'},
                        title='Association Rules: Confidence vs Lift'
                    )
                    fig.add_hline(y=1, line_dash="dash", line_color="gray")
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No rules with lift > 1 found (positive association)")
        else:
            st.info("No association rules found with current thresholds")

    with tab3:
        st.subheader("Merchant Affinity Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Same-Day Orders (Complementary)**")
            complementary = cross_seller.get_complementary_merchants()

            if len(complementary) > 0:
                st.caption("Merchants often ordered together on the same day")
                st.dataframe(
                    complementary.head(15).style.format({'same_day_orders': '{:,}'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No complementary merchant data")

        with col2:
            st.markdown("**Sequential Orders (Substitutes)**")
            substitutes = cross_seller.get_merchant_substitutes()

            if len(substitutes) > 0:
                st.caption("Merchants users switch between")
                st.dataframe(
                    substitutes.head(15).style.format({'switch_count': '{:,}'}),
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No substitute merchant data")

    with tab4:
        st.subheader("Upsell Opportunities")

        upsell = affinity.get_upsell_opportunities()

        if len(upsell) > 0:
            st.caption("Products often bought alone that could be bundled")

            st.dataframe(
                upsell.style.format({'single_orders': '{:,}'}),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("Not enough data for upsell analysis")

        # Product pair analysis
        st.subheader("Frequently Bought Together")
        pairs = affinity.get_product_pairs(min_support=2)

        if len(pairs) > 0:
            fig = px.bar(
                pairs.head(20),
                x='co_occurrence',
                y=pairs.head(20).apply(lambda r: f"{r['product_1'][:20]} + {r['product_2'][:20]}", axis=1),
                orientation='h',
                labels={'co_occurrence': 'Times Bought Together', 'y': 'Product Pair'},
                title='Top Product Pairs'
            )
            fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product pair data available")


if __name__ == '__main__':
    main()
