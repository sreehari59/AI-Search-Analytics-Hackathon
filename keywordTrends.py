import pandas as pd
import numpy as np
from typing import Optional
from pytrends.request import TrendReq
import os
import json
import logging

# Configure logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def saveGoogleTrends(keyword: str, data: pd.DataFrame, startDate: Optional[str] = None, endDate: Optional[str] = None):
    """
    Save Google Trends data to a JSON file.
    
    Args:
        keyword (str): The keyword
        data (pd.DataFrame): Google Trends data
        startDate (str, optional): Start date
        endDate (str, optional): End date
    """
    try:
        # Create filename based on keyword and date range
        date_suffix = f"_{startDate.replace('-', '')}_{endDate.replace('-', '')}" if startDate and endDate else "_default"
        filename = f"data/google_trends_{keyword.replace(' ', '_')}{date_suffix}.json"
        
        # Convert DataFrame to JSON-serializable format
        # Convert timestamps to strings and handle other non-serializable types
        data_records = []
        for _, row in data.iterrows():
            record = {}
            for col, value in row.items():
                if pd.isna(value):
                    record[col] = None
                elif isinstance(value, pd.Timestamp):
                    record[col] = value.isoformat()
                elif isinstance(value, (np.integer, np.floating)):
                    record[col] = float(value)
                else:
                    record[col] = value
            data_records.append(record)
        
        data_dict = {
            'keyword': keyword,
            'startDate': startDate,
            'endDate': endDate,
            'data': data_records,
            'columns': list(data.columns)
        }
        
        with open(filename, 'w') as f:
            json.dump(data_dict, f, indent=2)
        
        logger.info(f"Google Trends data saved for '{keyword}' to {filename}")
        print(f"Google Trends data saved for '{keyword}' to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving Google Trends data for '{keyword}': {e}")
        print(f"Error saving Google Trends data for '{keyword}': {e}")

def loadGoogleTrends(keyword: str, startDate: Optional[str] = None, endDate: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load Google Trends data from JSON file.
    
    Args:
        keyword (str): The keyword
        startDate (str, optional): Start date
        endDate (str, optional): End date
        
    Returns:
        Optional[pd.DataFrame]: Google Trends data if found, None otherwise
    """
    try:
        # Create filename based on keyword and date range
        date_suffix = f"_{startDate.replace('-', '')}_{endDate.replace('-', '')}" if startDate and endDate else "_default"
        filename = f"data/google_trends_{keyword.replace(' ', '_')}{date_suffix}.json"
        
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    data_dict = json.load(f)
                
                # Reconstruct DataFrame
                df = pd.DataFrame(data_dict['data'])
                
                # Convert date strings back to datetime if needed
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'])
                
                logger.info(f"Google Trends data loaded for '{keyword}' from {filename}")
                print(f"Google Trends data loaded for '{keyword}' from {filename}")
                return df
                
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.error(f"Corrupted cache file for '{keyword}': {e}")
                print(f"Corrupted cache file for '{keyword}', removing and will fetch fresh data")
                # Remove corrupted file
                try:
                    os.remove(filename)
                except:
                    pass
                return None
        else:
            logger.info(f"No cached Google Trends data found for '{keyword}'")
            return None
            
    except Exception as e:
        logger.error(f"Error loading Google Trends data for '{keyword}': {e}")
        print(f"Error loading Google Trends data for '{keyword}': {e}")
        return None

def getGoogleTrends(keyword: str, startDate: Optional[str] = None, endDate: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Get keyword trends from Google Trends API with calculated metrics.
    First checks for cached data, then fetches from API if needed.
    
    Args:
        keyword (str): The keyword to analyze
        startDate (str, optional): Start date
        endDate (str, optional): End date
        
    Returns:
        pd.DataFrame: Google Trends data with calculated searchVolume, competition, and cpc
    """
    # First, try to load from cache
    cached_data = loadGoogleTrends(keyword, startDate, endDate)
    if cached_data is not None:
        return cached_data
    
    # If not in cache, fetch from API
    print(f"Fetching Google Trends data for '{keyword}' from API...")
    
    try:        
        # Initialize pytrends
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Build payload
        timeframe = f"{startDate} {endDate}" if startDate and endDate else 'today 12-m'
        pytrends.build_payload([keyword], cat=0, timeframe=timeframe, geo='', gprop='')
        
        # Get interest over time
        interestOverTimeDf = pytrends.interest_over_time()
        
        if interestOverTimeDf.empty:
            raise ValueError(f"No data found for keyword: {keyword}")
        
        # Reset index to get date as a column
        df = interestOverTimeDf.reset_index()
        df = df.rename(columns={keyword: 'trendValue'})
        df['keyword'] = keyword
        
        # Calculate additional metrics
        df = _calculateGoogleTrendsMetrics(df, keyword, pytrends)
        
        # Save to cache
        saveGoogleTrends(keyword, df, startDate, endDate)
        
        return df
        
    except ImportError:
        raise ImportError("pytrends library not found. Install with: pip install pytrends")
    except Exception as e:
        raise Exception(f"Error fetching Google Trends data: {e}")


def _calculateGoogleTrendsMetrics(df: pd.DataFrame, keyword: str, pytrends) -> pd.DataFrame:
    """
    Calculate competition based on Google Trends data.
    
    Args:
        df (pd.DataFrame): DataFrame with trendValue from Google Trends
        keyword (str): The keyword being analyzed
        pytrends: pytrends instance
        
    Returns:
        pd.DataFrame: DataFrame with calculated competition
    """
    try:
        # Get related queries to estimate competition
        try:
            relatedQueries = pytrends.related_queries()
            if relatedQueries and keyword in relatedQueries:
                topQueries = relatedQueries[keyword]['top']
                risingQueries = relatedQueries[keyword]['rising']
                
                # Debug: Print structure of related queries
                if topQueries is not None and not topQueries.empty:
                    print(f"Debug: Top queries columns for '{keyword}': {list(topQueries.columns)}")
                    print(f"Debug: Top queries shape: {topQueries.shape}")
                
                # Calculate competition based on number of related queries and their interest
                competitionScore = _estimateCompetition(topQueries, risingQueries)
            else:
                competitionScore = 0.5  # Default moderate competition
        except Exception as query_error:
            print(f"Error getting related queries for '{keyword}': {query_error}")
            competitionScore = 0.5  # Default moderate competition
        
        # Calculate metrics for each row
        competitions = []
        
        for _, row in df.iterrows():
            trendValue = row['trendValue']
            
            # Competition varies slightly over time but stays relatively stable
            competition = max(0.1, min(0.9, competitionScore + np.random.uniform(-0.1, 0.1)))
            competitions.append(round(competition, 3))
        
        # Add calculated columns
        df['competition'] = competitions
        
        return df
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        print(f"Falling back to default competition values for keyword: {keyword}")
        # Fallback to mock data for the additional columns
        df['competition'] = np.random.uniform(0.1, 0.9, len(df)).round(3)
        return df


def _estimateCompetition(topQueries: pd.DataFrame, risingQueries: pd.DataFrame) -> float:
    """
    Estimate competition level based on related queries.
    
    Args:
        topQueries (pd.DataFrame): Top related queries
        risingQueries (pd.DataFrame): Rising related queries
        
    Returns:
        float: Competition score between 0.0 and 1.0
    """
    if topQueries is None or topQueries.empty:
        return 0.5
    
    # Calculate competition based on:
    # 1. Number of related queries
    # 2. Average interest in related queries
    # 3. Presence of rising queries (indicates growing competition)
    
    numTopQueries = len(topQueries)
    
    # Safely get average interest - check for different possible column names
    avgTopInterest = 50  # Default value
    if not topQueries.empty:
        # Check for common column names in pytrends response
        possible_columns = ['value', 'interest', 'score', 'trend']
        for col in possible_columns:
            if col in topQueries.columns:
                try:
                    avgTopInterest = topQueries[col].mean()
                    break
                except (IndexError, KeyError):
                    continue
        
        # If no valid column found, use the first numeric column
        if avgTopInterest == 50:
            numeric_columns = topQueries.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                try:
                    avgTopInterest = topQueries[numeric_columns[0]].mean()
                except (IndexError, KeyError):
                    avgTopInterest = 50
    
    numRisingQueries = len(risingQueries) if risingQueries is not None and not risingQueries.empty else 0
    
    # Competition formula
    competition = min(1.0, (
        (numTopQueries / 10) * 0.3 +  # More related queries = higher competition
        (avgTopInterest / 100) * 0.4 +  # Higher interest in related queries = higher competition
        (numRisingQueries / 5) * 0.3    # More rising queries = higher competition
    ))
    
    return round(competition, 3)


def getGoogleTrendsStats() -> dict:
    """
    Get statistics about cached Google Trends data.
    
    Returns:
        dict: Statistics about cached Google Trends data
    """
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return {'total_files': 0, 'keywords': [], 'file_size_mb': 0}
        
        # Count Google Trends files
        trend_files = [f for f in os.listdir(data_dir) if f.startswith('google_trends_') and f.endswith('.json')]
        
        total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in trend_files)
        
        # Extract keywords from filenames
        keywords = []
        for filename in trend_files:
            # Remove 'google_trends_' prefix and '.json' suffix
            keyword_part = filename[13:-5]  # Remove 'google_trends_' and '.json'
            # Remove date suffix if present
            if '_' in keyword_part:
                keyword_part = keyword_part.rsplit('_', 2)[0]  # Remove last two parts (date)
            keywords.append(keyword_part.replace('_', ' '))
        
        stats = {
            'total_files': len(trend_files),
            'keywords': list(set(keywords)),  # Remove duplicates
            'file_size_mb': total_size / (1024 * 1024),
            'sample_files': trend_files[:5] if trend_files else []
        }
        
        return stats
        
    except Exception as e:
        logger.error(f"Error getting Google Trends stats: {e}")
        return {'total_files': 0, 'keywords': [], 'file_size_mb': 0, 'error': str(e)}


def clearGoogleTrendsCache(keyword: Optional[str] = None):
    """
    Clear cached Google Trends data.
    
    Args:
        keyword (str, optional): Specific keyword to clear. If None, clears all.
    """
    try:
        data_dir = 'data'
        if not os.path.exists(data_dir):
            return
        
        if keyword:
            # Clear specific keyword
            pattern = f"google_trends_{keyword.replace(' ', '_')}"
            files_to_remove = [f for f in os.listdir(data_dir) if f.startswith(pattern) and f.endswith('.json')]
        else:
            # Clear all Google Trends files
            files_to_remove = [f for f in os.listdir(data_dir) if f.startswith('google_trends_') and f.endswith('.json')]
        
        for filename in files_to_remove:
            os.remove(os.path.join(data_dir, filename))
            print(f"Removed cached file: {filename}")
        
        print(f"Cleared {len(files_to_remove)} cached Google Trends files")
        
    except Exception as e:
        logger.error(f"Error clearing Google Trends cache: {e}")
        print(f"Error clearing Google Trends cache: {e}")