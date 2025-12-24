"""Product Affinity & Bundle Analysis for Snoonu ML Framework.

Implements market basket analysis to find:
- Products frequently bought together
- Cross-sell and upsell opportunities
- Bundle recommendations by merchant
- Association rules (support, confidence, lift)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from itertools import combinations
import json


class ProductAffinityAnalyzer:
    """Analyze product co-purchase patterns and generate bundle recommendations."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with event data."""
        self.df = df.copy()
        self.transactions = None
        self._prepare_data()

    def _prepare_data(self):
        """Prepare transaction data from events."""
        # Get checkout events with product info
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        if orders.empty:
            self.transactions = pd.DataFrame()
            return

        # Parse order properties
        transactions = []

        for _, order in orders.iterrows():
            try:
                props = order.get('event_properties', '{}')
                if isinstance(props, str):
                    props = json.loads(props)

                order_id = props.get('order_id', f"order_{order.name}")
                merchant_id = props.get('merchant_id', 'unknown')
                merchant_name = props.get('merchant_name', 'Unknown Merchant')

                # Get products - they may be in different formats
                products = props.get('products', [])
                product_names = props.get('product_names', [])

                if isinstance(products, list) and products:
                    for product in products:
                        if isinstance(product, dict):
                            transactions.append({
                                'order_id': order_id,
                                'user_id': order['amplitude_id'],
                                'merchant_id': merchant_id,
                                'merchant_name': merchant_name,
                                'product_id': product.get('product_id', product.get('id', 'unknown')),
                                'product_name': product.get('product_name', product.get('name', 'Unknown')),
                                'quantity': product.get('quantity', 1),
                                'price': product.get('price', 0)
                            })
                        elif isinstance(product, str):
                            transactions.append({
                                'order_id': order_id,
                                'user_id': order['amplitude_id'],
                                'merchant_id': merchant_id,
                                'merchant_name': merchant_name,
                                'product_id': product,
                                'product_name': product,
                                'quantity': 1,
                                'price': 0
                            })
                elif isinstance(product_names, list) and product_names:
                    for pname in product_names:
                        transactions.append({
                            'order_id': order_id,
                            'user_id': order['amplitude_id'],
                            'merchant_id': merchant_id,
                            'merchant_name': merchant_name,
                            'product_id': pname,
                            'product_name': pname,
                            'quantity': 1,
                            'price': 0
                        })
                else:
                    # No product details available, track at order level
                    transactions.append({
                        'order_id': order_id,
                        'user_id': order['amplitude_id'],
                        'merchant_id': merchant_id,
                        'merchant_name': merchant_name,
                        'product_id': f"order_{order_id}",
                        'product_name': f"Order from {merchant_name}",
                        'quantity': props.get('products_quantity', 1),
                        'price': props.get('order_total', 0)
                    })

            except Exception as e:
                continue

        self.transactions = pd.DataFrame(transactions)

    def get_merchant_affinity(self) -> pd.DataFrame:
        """Find merchants frequently ordered from together (multi-cart orders)."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        if orders.empty:
            return pd.DataFrame()

        # Group by user and day to find same-day orders from different merchants
        orders['order_date'] = pd.to_datetime(orders['event_time']).dt.date

        merchant_pairs = defaultdict(int)
        user_merchant_groups = orders.groupby(['amplitude_id', 'order_date'])

        for (user_id, order_date), group in user_merchant_groups:
            if len(group) < 2:
                continue

            # Parse merchant IDs
            merchants = set()
            for _, order in group.iterrows():
                try:
                    props = order.get('event_properties', '{}')
                    if isinstance(props, str):
                        props = json.loads(props)
                    merchant = props.get('merchant_name', props.get('merchant_id', 'unknown'))
                    if merchant and merchant != 'unknown':
                        merchants.add(merchant)
                except:
                    continue

            # Create pairs
            for pair in combinations(sorted(merchants), 2):
                merchant_pairs[pair] += 1

        # Convert to DataFrame
        pairs_df = pd.DataFrame([
            {
                'merchant_1': pair[0],
                'merchant_2': pair[1],
                'co_occurrence': count
            }
            for pair, count in merchant_pairs.items()
        ])

        if pairs_df.empty:
            return pd.DataFrame()

        return pairs_df.sort_values('co_occurrence', ascending=False)

    def get_product_pairs(self, min_support: int = 2, merchant_id: str = None) -> pd.DataFrame:
        """Find product pairs frequently bought together."""
        if self.transactions.empty:
            return pd.DataFrame()

        # Filter by merchant if specified
        trans = self.transactions
        if merchant_id:
            trans = trans[trans['merchant_id'] == merchant_id]

        # Group products by order
        order_products = trans.groupby('order_id')['product_name'].apply(set).reset_index()

        # Count pair occurrences
        pair_counts = defaultdict(int)
        total_orders = len(order_products)

        for _, row in order_products.iterrows():
            products = row['product_name']
            if len(products) >= 2:
                for pair in combinations(sorted(products), 2):
                    pair_counts[pair] += 1

        # Filter by minimum support
        pairs_df = pd.DataFrame([
            {
                'product_1': pair[0],
                'product_2': pair[1],
                'co_occurrence': count,
                'support': count / total_orders * 100
            }
            for pair, count in pair_counts.items()
            if count >= min_support
        ])

        if pairs_df.empty:
            return pd.DataFrame()

        return pairs_df.sort_values('co_occurrence', ascending=False)

    def calculate_association_rules(self, min_support: float = 0.01, min_confidence: float = 0.1) -> pd.DataFrame:
        """Calculate association rules with support, confidence, and lift."""
        if self.transactions.empty:
            return pd.DataFrame()

        # Group products by order
        order_products = self.transactions.groupby('order_id')['product_name'].apply(set).reset_index()
        total_orders = len(order_products)

        if total_orders < 10:
            return pd.DataFrame()

        # Calculate individual product frequencies
        product_freq = defaultdict(int)
        for _, row in order_products.iterrows():
            for product in row['product_name']:
                product_freq[product] += 1

        # Calculate pair frequencies
        pair_freq = defaultdict(int)
        for _, row in order_products.iterrows():
            products = row['product_name']
            if len(products) >= 2:
                for pair in combinations(products, 2):
                    pair_freq[frozenset(pair)] += 1

        # Generate rules
        rules = []

        for pair, pair_count in pair_freq.items():
            pair_support = pair_count / total_orders
            if pair_support < min_support:
                continue

            items = list(pair)

            # Rule: A -> B
            support_a = product_freq[items[0]] / total_orders
            confidence_ab = pair_count / product_freq[items[0]]
            support_b = product_freq[items[1]] / total_orders
            lift_ab = confidence_ab / support_b if support_b > 0 else 0

            if confidence_ab >= min_confidence:
                rules.append({
                    'antecedent': items[0],
                    'consequent': items[1],
                    'support': pair_support * 100,
                    'confidence': confidence_ab * 100,
                    'lift': lift_ab,
                    'count': pair_count
                })

            # Rule: B -> A
            confidence_ba = pair_count / product_freq[items[1]]
            lift_ba = confidence_ba / support_a if support_a > 0 else 0

            if confidence_ba >= min_confidence:
                rules.append({
                    'antecedent': items[1],
                    'consequent': items[0],
                    'support': pair_support * 100,
                    'confidence': confidence_ba * 100,
                    'lift': lift_ba,
                    'count': pair_count
                })

        rules_df = pd.DataFrame(rules)

        if rules_df.empty:
            return pd.DataFrame()

        return rules_df.sort_values('lift', ascending=False)

    def get_bundle_recommendations(self, top_n: int = 10) -> List[Dict]:
        """Generate bundle recommendations based on co-purchase patterns."""
        rules = self.calculate_association_rules()

        if rules.empty:
            return []

        # Filter for high-lift rules (lift > 1 means positive association)
        strong_rules = rules[rules['lift'] > 1.5].head(top_n * 2)

        bundles = []
        seen_pairs = set()

        for _, rule in strong_rules.iterrows():
            pair_key = tuple(sorted([rule['antecedent'], rule['consequent']]))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            bundles.append({
                'product_1': rule['antecedent'],
                'product_2': rule['consequent'],
                'lift': rule['lift'],
                'confidence': rule['confidence'],
                'recommendation': f"Customers who buy '{rule['antecedent']}' often also buy '{rule['consequent']}'",
                'bundle_strength': 'strong' if rule['lift'] > 2 else 'moderate'
            })

            if len(bundles) >= top_n:
                break

        return bundles

    def get_cross_sell_recommendations(self, product_name: str, top_n: int = 5) -> List[Dict]:
        """Get cross-sell recommendations for a specific product."""
        rules = self.calculate_association_rules()

        if rules.empty:
            return []

        # Find rules where the product is the antecedent
        product_rules = rules[
            rules['antecedent'].str.lower() == product_name.lower()
        ].head(top_n)

        recommendations = []
        for _, rule in product_rules.iterrows():
            recommendations.append({
                'recommended_product': rule['consequent'],
                'confidence': rule['confidence'],
                'lift': rule['lift'],
                'reason': f"{rule['confidence']:.0f}% of customers who bought this also bought {rule['consequent']}"
            })

        return recommendations

    def get_merchant_top_bundles(self) -> pd.DataFrame:
        """Get top product bundles by merchant."""
        if self.transactions.empty:
            return pd.DataFrame()

        merchants = self.transactions['merchant_name'].unique()
        merchant_bundles = []

        for merchant in merchants:
            merchant_trans = self.transactions[self.transactions['merchant_name'] == merchant]

            # Group by order
            order_products = merchant_trans.groupby('order_id')['product_name'].apply(list).reset_index()

            # Find most common pair
            pair_counts = defaultdict(int)
            for _, row in order_products.iterrows():
                products = row['product_name']
                if len(products) >= 2:
                    for pair in combinations(sorted(set(products)), 2):
                        pair_counts[pair] += 1

            if pair_counts:
                top_pair = max(pair_counts.items(), key=lambda x: x[1])
                merchant_bundles.append({
                    'merchant': merchant,
                    'product_1': top_pair[0][0],
                    'product_2': top_pair[0][1],
                    'co_occurrence': top_pair[1],
                    'total_orders': len(order_products)
                })

        bundles_df = pd.DataFrame(merchant_bundles)

        if bundles_df.empty:
            return pd.DataFrame()

        return bundles_df.sort_values('co_occurrence', ascending=False)

    def get_upsell_opportunities(self) -> pd.DataFrame:
        """Identify upsell opportunities based on order patterns."""
        if self.transactions.empty:
            return pd.DataFrame()

        # Group by order to calculate basket size
        order_stats = self.transactions.groupby('order_id').agg({
            'product_name': 'count',
            'price': 'sum',
            'merchant_name': 'first'
        }).reset_index()
        order_stats.columns = ['order_id', 'items_count', 'order_total', 'merchant']

        # Find single-item orders
        single_item_orders = order_stats[order_stats['items_count'] == 1]

        # Get products commonly bought alone
        single_products = self.transactions[
            self.transactions['order_id'].isin(single_item_orders['order_id'])
        ]['product_name'].value_counts().head(10)

        # For each, find what's commonly added
        upsell_opps = []

        for product in single_products.index:
            # Find orders with this product that have 2+ items
            multi_item_with_product = self.transactions[
                (self.transactions['product_name'] == product) &
                (self.transactions['order_id'].isin(
                    order_stats[order_stats['items_count'] > 1]['order_id']
                ))
            ]['order_id'].unique()

            # Find commonly co-purchased items
            co_products = self.transactions[
                (self.transactions['order_id'].isin(multi_item_with_product)) &
                (self.transactions['product_name'] != product)
            ]['product_name'].value_counts().head(3)

            if not co_products.empty:
                upsell_opps.append({
                    'base_product': product,
                    'single_orders': single_products[product],
                    'upsell_suggestion_1': co_products.index[0] if len(co_products) > 0 else None,
                    'upsell_suggestion_2': co_products.index[1] if len(co_products) > 1 else None,
                    'upsell_suggestion_3': co_products.index[2] if len(co_products) > 2 else None
                })

        return pd.DataFrame(upsell_opps)

    def get_affinity_summary(self) -> Dict:
        """Get summary of product affinity analysis."""
        if self.transactions.empty:
            return {
                'total_transactions': 0,
                'unique_products': 0,
                'avg_basket_size': 0,
                'top_bundles': []
            }

        order_sizes = self.transactions.groupby('order_id')['product_name'].count()

        bundles = self.get_bundle_recommendations(top_n=5)

        return {
            'total_transactions': self.transactions['order_id'].nunique(),
            'unique_products': self.transactions['product_name'].nunique(),
            'unique_merchants': self.transactions['merchant_name'].nunique(),
            'avg_basket_size': order_sizes.mean(),
            'max_basket_size': order_sizes.max(),
            'multi_item_order_rate': (order_sizes > 1).sum() / len(order_sizes) * 100,
            'top_bundles': bundles
        }


class MerchantCrossSeller:
    """Analyze cross-merchant ordering patterns."""

    def __init__(self, df: pd.DataFrame):
        """Initialize with event data."""
        self.df = df.copy()

    def get_merchant_substitutes(self) -> pd.DataFrame:
        """Find merchants that users switch between (potential substitutes)."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        if orders.empty:
            return pd.DataFrame()

        # Parse merchant info
        order_merchants = []
        for _, order in orders.iterrows():
            try:
                props = order.get('event_properties', '{}')
                if isinstance(props, str):
                    props = json.loads(props)
                merchant = props.get('merchant_name', props.get('merchant_id'))
                if merchant:
                    order_merchants.append({
                        'user_id': order['amplitude_id'],
                        'merchant': merchant,
                        'order_time': order['event_time']
                    })
            except:
                continue

        orders_df = pd.DataFrame(order_merchants)

        if orders_df.empty:
            return pd.DataFrame()

        orders_df = orders_df.sort_values(['user_id', 'order_time'])

        # Find sequential merchant switches
        switch_counts = defaultdict(int)
        user_groups = orders_df.groupby('user_id')

        for user_id, group in user_groups:
            merchants = group['merchant'].tolist()
            for i in range(len(merchants) - 1):
                if merchants[i] != merchants[i + 1]:
                    pair = tuple(sorted([merchants[i], merchants[i + 1]]))
                    switch_counts[pair] += 1

        switches_df = pd.DataFrame([
            {
                'merchant_1': pair[0],
                'merchant_2': pair[1],
                'switch_count': count
            }
            for pair, count in switch_counts.items()
        ])

        if switches_df.empty:
            return pd.DataFrame()

        return switches_df.sort_values('switch_count', ascending=False).head(20)

    def get_complementary_merchants(self) -> pd.DataFrame:
        """Find merchants ordered from on the same day (complementary)."""
        orders = self.df[self.df['event_type'] == 'checkout_completed'].copy()

        if orders.empty:
            return pd.DataFrame()

        orders['order_date'] = pd.to_datetime(orders['event_time']).dt.date

        # Parse and group
        same_day = defaultdict(int)

        for (user_id, order_date), group in orders.groupby(['amplitude_id', 'order_date']):
            merchants = set()
            for _, order in group.iterrows():
                try:
                    props = order.get('event_properties', '{}')
                    if isinstance(props, str):
                        props = json.loads(props)
                    merchant = props.get('merchant_name', props.get('merchant_id'))
                    if merchant:
                        merchants.add(merchant)
                except:
                    continue

            if len(merchants) >= 2:
                for pair in combinations(sorted(merchants), 2):
                    same_day[pair] += 1

        comp_df = pd.DataFrame([
            {
                'merchant_1': pair[0],
                'merchant_2': pair[1],
                'same_day_orders': count
            }
            for pair, count in same_day.items()
        ])

        if comp_df.empty:
            return pd.DataFrame()

        return comp_df.sort_values('same_day_orders', ascending=False).head(20)
