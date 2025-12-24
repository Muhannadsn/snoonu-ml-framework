"""Search Analytics.

Analysis of user search behavior, conversion, and optimization opportunities.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import Counter
import json
import re
import logging

logger = logging.getLogger(__name__)


class SearchAnalyzer:
    """Comprehensive search behavior analytics."""

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: Event data DataFrame with search_made events
        """
        self.df = df.copy()
        self._extract_search_data()

    def _extract_search_data(self):
        """Extract and parse search data."""
        logger.info("Extracting search data...")

        # Get search events
        searches = self.df[self.df['event_type'] == 'search_made'].copy()

        if len(searches) == 0:
            logger.warning("No search_made events found")
            self.searches = pd.DataFrame()
            self.has_data = False
            return

        # Parse search properties
        def parse_props(props):
            if isinstance(props, str):
                try:
                    return json.loads(props)
                except:
                    return {}
            return props if isinstance(props, dict) else {}

        if 'event_properties' in searches.columns:
            searches['props'] = searches['event_properties'].apply(parse_props)
            searches['search_query'] = searches['props'].apply(
                lambda x: x.get('search_query') or x.get('query') or x.get('search_term', '')
            )
            searches['results_count'] = searches['props'].apply(
                lambda x: int(x.get('results_count', 0) or x.get('num_results', 0) or 0)
            )
            searches['search_type'] = searches['props'].apply(
                lambda x: x.get('search_type', 'text')
            )
        elif 'event_data_search_query' in searches.columns:
            searches['search_query'] = searches['event_data_search_query']
            searches['results_count'] = searches.get('event_data_results_count', 0)
            searches['search_type'] = 'text'
        else:
            # Try to extract from any available column
            searches['search_query'] = ''
            searches['results_count'] = 0
            searches['search_type'] = 'text'

        searches['event_time'] = pd.to_datetime(searches['event_time'])
        searches['hour'] = searches['event_time'].dt.hour
        searches['date'] = searches['event_time'].dt.date

        # Clean search queries
        searches['search_query'] = searches['search_query'].fillna('').astype(str).str.strip().str.lower()
        searches = searches[searches['search_query'] != '']

        # Flag zero-result searches
        searches['zero_results'] = searches['results_count'] == 0

        self.searches = searches
        self.has_data = len(self.searches) > 0

        # Get conversion data (users who searched and then ordered)
        searchers = set(self.searches['amplitude_id'])
        checkouts = self.df[self.df['event_type'] == 'checkout_completed']
        self.searcher_orders = checkouts[checkouts['amplitude_id'].isin(searchers)]

        logger.info(f"Extracted {len(self.searches):,} searches from {self.searches['amplitude_id'].nunique():,} users")

    def get_summary(self) -> Dict:
        """Get search analytics summary."""
        if not self.has_data:
            return {'error': 'No search data available'}

        total_searches = len(self.searches)
        unique_searchers = self.searches['amplitude_id'].nunique()
        unique_queries = self.searches['search_query'].nunique()
        zero_result_rate = self.searches['zero_results'].mean() * 100

        # Search to order conversion
        searchers_who_ordered = self.searcher_orders['amplitude_id'].nunique()
        search_conversion = searchers_who_ordered / unique_searchers * 100 if unique_searchers > 0 else 0

        return {
            'total_searches': total_searches,
            'unique_searchers': unique_searchers,
            'unique_queries': unique_queries,
            'searches_per_user': round(total_searches / unique_searchers, 2) if unique_searchers > 0 else 0,
            'zero_result_rate': round(zero_result_rate, 1),
            'search_to_order_rate': round(search_conversion, 1),
            'avg_results_per_search': round(self.searches['results_count'].mean(), 1)
        }

    def get_top_searches(self, n: int = 50) -> pd.DataFrame:
        """Get most popular search queries."""
        if not self.has_data:
            return pd.DataFrame()

        query_stats = self.searches.groupby('search_query').agg({
            'amplitude_id': ['count', 'nunique'],
            'zero_results': 'mean',
            'results_count': 'mean'
        }).reset_index()

        query_stats.columns = ['query', 'searches', 'unique_users', 'zero_result_rate', 'avg_results']
        query_stats['zero_result_rate'] = (query_stats['zero_result_rate'] * 100).round(1)
        query_stats['avg_results'] = query_stats['avg_results'].round(1)

        # Check conversion for each query
        def get_conversion(query):
            query_users = set(self.searches[self.searches['search_query'] == query]['amplitude_id'])
            ordered = self.searcher_orders[self.searcher_orders['amplitude_id'].isin(query_users)]['amplitude_id'].nunique()
            return ordered / len(query_users) * 100 if query_users else 0

        query_stats['conversion_rate'] = query_stats['query'].apply(get_conversion).round(1)

        query_stats = query_stats.sort_values('searches', ascending=False).head(n)
        query_stats['rank'] = range(1, len(query_stats) + 1)

        return query_stats[['rank', 'query', 'searches', 'unique_users', 'avg_results', 'zero_result_rate', 'conversion_rate']]

    def get_zero_result_searches(self, n: int = 30) -> pd.DataFrame:
        """Get searches that returned zero results - optimization opportunities."""
        if not self.has_data:
            return pd.DataFrame()

        zero_results = self.searches[self.searches['zero_results']]

        if len(zero_results) == 0:
            return pd.DataFrame({'message': ['No zero-result searches found']})

        zero_stats = zero_results.groupby('search_query').agg({
            'amplitude_id': ['count', 'nunique']
        }).reset_index()

        zero_stats.columns = ['query', 'occurrences', 'unique_users']
        zero_stats = zero_stats.sort_values('occurrences', ascending=False).head(n)
        zero_stats['priority'] = pd.cut(
            zero_stats['occurrences'],
            bins=[0, 5, 20, 100, float('inf')],
            labels=['Low', 'Medium', 'High', 'Critical']
        )

        return zero_stats

    def get_search_conversion_funnel(self) -> pd.DataFrame:
        """Get search-to-order conversion funnel."""
        if not self.has_data:
            return pd.DataFrame()

        searchers = set(self.searches['amplitude_id'])

        # Track through funnel
        funnel_events = ['search_made', 'product_page_viewed', 'product_added', 'checkout_completed']
        funnel_data = []

        for event in funnel_events:
            event_users = set(self.df[self.df['event_type'] == event]['amplitude_id'])
            searcher_users = searchers & event_users
            funnel_data.append({
                'step': event.replace('_', ' ').title(),
                'searchers': len(searcher_users),
                'pct_of_searchers': len(searcher_users) / len(searchers) * 100 if searchers else 0
            })

        return pd.DataFrame(funnel_data)

    def get_search_by_hour(self) -> pd.DataFrame:
        """Get search volume and conversion by hour."""
        if not self.has_data:
            return pd.DataFrame()

        hourly = self.searches.groupby('hour').agg({
            'amplitude_id': ['count', 'nunique'],
            'zero_results': 'mean'
        }).reset_index()

        hourly.columns = ['hour', 'searches', 'unique_searchers', 'zero_result_rate']
        hourly['zero_result_rate'] = (hourly['zero_result_rate'] * 100).round(1)
        hourly['searches_per_user'] = (hourly['searches'] / hourly['unique_searchers']).round(2)

        return hourly

    def get_search_categories(self) -> pd.DataFrame:
        """Categorize searches by intent/type."""
        if not self.has_data:
            return pd.DataFrame()

        # Common food categories to match
        categories = {
            'Fast Food': ['burger', 'pizza', 'fries', 'kfc', 'mcdonald', 'subway', 'wendys', 'chicken'],
            'Asian': ['sushi', 'chinese', 'thai', 'indian', 'japanese', 'korean', 'noodle', 'ramen', 'curry'],
            'Arabic/Middle Eastern': ['shawarma', 'kebab', 'falafel', 'hummus', 'arabic', 'lebanese', 'turkish'],
            'Desserts': ['cake', 'ice cream', 'dessert', 'chocolate', 'sweet', 'donut', 'bakery'],
            'Coffee/Drinks': ['coffee', 'tea', 'juice', 'smoothie', 'starbucks', 'cafe'],
            'Healthy': ['salad', 'healthy', 'vegan', 'vegetarian', 'organic', 'diet'],
            'Grocery': ['grocery', 'supermarket', 'snacks', 'water', 'milk']
        }

        def categorize_query(query):
            query_lower = query.lower()
            for category, keywords in categories.items():
                if any(kw in query_lower for kw in keywords):
                    return category
            return 'Other'

        self.searches['category'] = self.searches['search_query'].apply(categorize_query)

        category_stats = self.searches.groupby('category').agg({
            'amplitude_id': ['count', 'nunique'],
            'zero_results': 'mean'
        }).reset_index()

        category_stats.columns = ['category', 'searches', 'unique_users', 'zero_result_rate']
        category_stats['zero_result_rate'] = (category_stats['zero_result_rate'] * 100).round(1)
        category_stats['pct_of_searches'] = (category_stats['searches'] / category_stats['searches'].sum() * 100).round(1)
        category_stats = category_stats.sort_values('searches', ascending=False)

        return category_stats

    def get_search_refinements(self) -> pd.DataFrame:
        """Analyze search refinements (users who search multiple times in a session)."""
        if not self.has_data:
            return pd.DataFrame()

        # Group searches by user within short time windows (10 min = same search session)
        self.searches = self.searches.sort_values(['amplitude_id', 'event_time'])
        self.searches['prev_time'] = self.searches.groupby('amplitude_id')['event_time'].shift(1)
        self.searches['time_gap'] = (self.searches['event_time'] - self.searches['prev_time']).dt.total_seconds() / 60

        # Same search session if gap < 10 minutes
        self.searches['new_search_session'] = (
            self.searches['time_gap'].isna() | (self.searches['time_gap'] > 10)
        ).astype(int)
        self.searches['search_session'] = self.searches.groupby('amplitude_id')['new_search_session'].cumsum()

        # Count searches per session
        session_searches = self.searches.groupby(['amplitude_id', 'search_session']).agg({
            'search_query': ['count', lambda x: list(x)]
        }).reset_index()
        session_searches.columns = ['amplitude_id', 'search_session', 'search_count', 'queries']

        # Refinement = 2+ searches in a session
        refinement_stats = session_searches.groupby('search_count').size().reset_index(name='sessions')
        refinement_stats['pct'] = (refinement_stats['sessions'] / refinement_stats['sessions'].sum() * 100).round(1)

        # Get common refinement patterns
        multi_search = session_searches[session_searches['search_count'] >= 2]
        refinement_rate = len(multi_search) / len(session_searches) * 100 if len(session_searches) > 0 else 0

        return refinement_stats, round(refinement_rate, 1)

    def get_search_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations based on search analysis."""
        if not self.has_data:
            return [{'type': 'info', 'title': 'No Data', 'message': 'No search data available for analysis'}]

        recommendations = []
        summary = self.get_summary()

        # High zero-result rate
        if summary['zero_result_rate'] > 10:
            zero_results = self.get_zero_result_searches(10)
            if len(zero_results) > 0 and 'query' in zero_results.columns:
                top_zero = zero_results.head(3)['query'].tolist()
                recommendations.append({
                    'type': 'warning',
                    'title': f"High Zero-Result Rate ({summary['zero_result_rate']:.1f}%)",
                    'message': f"Top failed searches: {', '.join(top_zero)}",
                    'action': 'Add these items/merchants or improve search matching'
                })

        # Low search-to-order conversion
        if summary['search_to_order_rate'] < 30:
            recommendations.append({
                'type': 'opportunity',
                'title': f"Low Search Conversion ({summary['search_to_order_rate']:.1f}%)",
                'message': 'Users who search are not converting to orders',
                'action': 'Improve search relevance, add filters, show popular items'
            })

        # High search volume queries with low conversion
        top_searches = self.get_top_searches(20)
        if len(top_searches) > 0:
            low_converting = top_searches[
                (top_searches['searches'] >= 10) &
                (top_searches['conversion_rate'] < 10)
            ]
            if len(low_converting) > 0:
                recommendations.append({
                    'type': 'opportunity',
                    'title': 'High-Volume Low-Converting Searches',
                    'message': f"{len(low_converting)} popular searches have <10% conversion",
                    'action': 'Review search results quality for these terms'
                })

        return recommendations
