"""
Data Loader Module
==================
Handles loading, validation, and preprocessing of Snoonu event data.
Supports parquet and CSV files with automatic schema validation.
"""

import pandas as pd
import pyarrow.parquet as pq
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load and validate Snoonu event data.

    Usage:
        loader = DataLoader(config_path='config.yaml')
        df = loader.load('data/dec_15.parquet')
    """

    def __init__(self, config_path: str = 'config.yaml'):
        """Initialize with configuration file."""
        self.config = self._load_config(config_path)
        self.schema = self.config.get('schema', {})
        self.events = self.config.get('events', {})

    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration file."""
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return {}

        with open(config_file, 'r') as f:
            return yaml.safe_load(f)

    def load(self, data_path: str, validate: bool = True) -> pd.DataFrame:
        """
        Load data from parquet or CSV file.

        Args:
            data_path: Path to data file
            validate: Whether to validate schema

        Returns:
            DataFrame with loaded data
        """
        path = Path(data_path)

        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        logger.info(f"Loading data from {data_path}")

        # Load based on file type
        if path.suffix == '.parquet':
            df = pq.read_table(data_path).to_pandas()
        elif path.suffix == '.csv':
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"Unsupported file type: {path.suffix}")

        logger.info(f"Loaded {len(df):,} rows, {len(df.columns)} columns")

        # Validate schema
        if validate:
            self._validate_schema(df)

        # Basic preprocessing
        df = self._preprocess(df)

        return df

    def load_multiple(self, file_paths: List[str], validate: bool = True) -> pd.DataFrame:
        """
        Load and combine multiple data files.

        Args:
            file_paths: List of paths to data files
            validate: Whether to validate schema

        Returns:
            Combined DataFrame with data from all files
        """
        if not file_paths:
            raise ValueError("No files provided")

        dfs = []
        for path in file_paths:
            try:
                df = self.load(path, validate=validate)
                # Add source file info
                df['_source_file'] = Path(path).name
                dfs.append(df)
                logger.info(f"Loaded {len(df):,} rows from {path}")
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        if not dfs:
            raise ValueError("No files could be loaded")

        # Combine all DataFrames
        combined = pd.concat(dfs, ignore_index=True)

        # Sort by user and time
        if 'amplitude_id' in combined.columns and 'event_time' in combined.columns:
            combined = combined.sort_values(['amplitude_id', 'event_time'])

        logger.info(f"Combined {len(file_paths)} files: {len(combined):,} total rows")

        return combined

    def load_date_range(self, folder_path: str, start_date: str = None,
                        end_date: str = None, pattern: str = "*.parquet") -> pd.DataFrame:
        """
        Load all files from a folder, optionally filtering by date in filename.

        Args:
            folder_path: Path to folder containing data files
            start_date: Start date string (e.g., "2024-12-09")
            end_date: End date string (e.g., "2024-12-15")
            pattern: Glob pattern for files (default: *.parquet)

        Returns:
            Combined DataFrame
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        # Find all matching files
        files = sorted(folder.glob(pattern))

        if not files:
            raise FileNotFoundError(f"No files matching {pattern} in {folder_path}")

        logger.info(f"Found {len(files)} files in {folder_path}")

        # Load all files
        return self.load_multiple([str(f) for f in files])

    def load_folder(self, folder_path: str, pattern: str = "*.parquet") -> pd.DataFrame:
        """
        Load all parquet/csv files from a folder.

        Args:
            folder_path: Path to folder
            pattern: Glob pattern (default: *.parquet)

        Returns:
            Combined DataFrame from all files
        """
        return self.load_date_range(folder_path, pattern=pattern)

    def _validate_schema(self, df: pd.DataFrame) -> None:
        """Validate that required columns exist."""
        required = self.schema.get('required_columns', [])
        optional = self.schema.get('optional_columns', [])

        # Check required columns
        missing_required = [col for col in required if col not in df.columns]
        if missing_required:
            raise ValueError(f"Missing required columns: {missing_required}")

        # Warn about missing optional columns
        missing_optional = [col for col in optional if col not in df.columns]
        if missing_optional:
            logger.warning(f"Missing optional columns: {missing_optional}")

        logger.info("Schema validation passed")

    def _preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Basic preprocessing steps."""
        # Ensure event_time is datetime
        if 'event_time' in df.columns:
            df['event_time'] = pd.to_datetime(df['event_time'])

        # Normalize platform
        if 'platform' in df.columns:
            df['platform'] = df['platform'].str.lower()

        # Sort by user and time
        if 'amplitude_id' in df.columns and 'event_time' in df.columns:
            df = df.sort_values(['amplitude_id', 'event_time'])

        return df

    def parse_event_properties(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse JSON event_properties column into a dict column.

        Args:
            df: DataFrame with event_properties column

        Returns:
            DataFrame with 'props' column containing parsed dicts
        """
        def safe_parse(x):
            if pd.isna(x):
                return {}
            try:
                return json.loads(x)
            except (json.JSONDecodeError, TypeError):
                return {}

        df = df.copy()
        df['props'] = df['event_properties'].apply(safe_parse)
        return df

    def extract_property(self, df: pd.DataFrame, property_name: str,
                         default=None) -> pd.Series:
        """
        Extract a specific property from event_properties.

        Args:
            df: DataFrame with 'props' column (from parse_event_properties)
            property_name: Name of property to extract
            default: Default value if property missing

        Returns:
            Series with extracted property values
        """
        if 'props' not in df.columns:
            df = self.parse_event_properties(df)

        return df['props'].apply(lambda x: x.get(property_name, default))

    def filter_events(self, df: pd.DataFrame,
                      event_types: List[str]) -> pd.DataFrame:
        """Filter DataFrame to specific event types."""
        return df[df['event_type'].isin(event_types)]

    def get_funnel_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get only funnel events (homepage to checkout)."""
        funnel = self.events.get('funnel', [])
        return self.filter_events(df, funnel)

    def get_order_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get only order-related events."""
        order_events = self.events.get('order_events', ['checkout_completed'])
        return self.filter_events(df, order_events)

    def get_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics of loaded data.

        Returns:
            Dict with summary stats
        """
        summary = {
            'total_events': len(df),
            'unique_users': df['amplitude_id'].nunique() if 'amplitude_id' in df.columns else None,
            'date_range': {
                'start': df['event_time'].min() if 'event_time' in df.columns else None,
                'end': df['event_time'].max() if 'event_time' in df.columns else None,
            },
            'event_types': df['event_type'].value_counts().to_dict() if 'event_type' in df.columns else None,
            'platforms': df['platform'].value_counts().to_dict() if 'platform' in df.columns else None,
        }
        return summary

    def print_summary(self, df: pd.DataFrame) -> None:
        """Print a formatted summary of the data."""
        summary = self.get_summary(df)

        print("\n" + "=" * 60)
        print("DATA SUMMARY")
        print("=" * 60)
        print(f"Total events: {summary['total_events']:,}")
        print(f"Unique users: {summary['unique_users']:,}")
        print(f"Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")

        print("\nEvent types:")
        if summary['event_types']:
            for event, count in sorted(summary['event_types'].items(),
                                       key=lambda x: -x[1]):
                print(f"  {event}: {count:,}")

        print("\nPlatforms:")
        if summary['platforms']:
            for platform, count in summary['platforms'].items():
                print(f"  {platform}: {count:,}")

        print("=" * 60 + "\n")


# Convenience function for quick loading
def load_data(data_path: str, config_path: str = 'config.yaml') -> pd.DataFrame:
    """
    Quick function to load data.

    Usage:
        from data_loader import load_data
        df = load_data('data/dec_15.parquet')
    """
    loader = DataLoader(config_path)
    return loader.load(data_path)


if __name__ == '__main__':
    # Test the loader
    import sys

    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = '/Users/muhannadsaad/Desktop/investigation/dec_15/dec_15_25.parquet'

    loader = DataLoader()
    df = loader.load(data_path)
    loader.print_summary(df)
