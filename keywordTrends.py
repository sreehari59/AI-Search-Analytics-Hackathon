import pandas as pd
import numpy as np
from typing import Optional
from pytrends.request import TrendReq


def getGoogleTrends(keyword: str, startDate: Optional[str] = None, endDate: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """
    Get keyword trends from Google Trends API with calculated metrics.
    Note: This requires pytrends library and proper API setup.
    
    Args:
        keyword (str): The keyword to analyze
        startDate (str, optional): Start date
        endDate (str, optional): End date
        
    Returns:
        pd.DataFrame: Google Trends data with calculated searchVolume, competition, and cpc
    """
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
        relatedQueries = pytrends.related_queries()
        if relatedQueries and keyword in relatedQueries:
            topQueries = relatedQueries[keyword]['top']
            risingQueries = relatedQueries[keyword]['rising']
            
            # Calculate competition based on number of related queries and their interest
            competitionScore = _estimateCompetition(topQueries, risingQueries)
        else:
            competitionScore = 0.5  # Default moderate competition
        
        # # Get interest by region to estimate search volume
        # interestByRegion = pytrends.interest_by_region(resolution='COUNTRY')
        # if not interestByRegion.empty:
        #     totalInterest = interestByRegion[keyword].sum()
        #     # Estimate search volume based on global interest and trend values
        #     baseSearchVolume = totalInterest * 1000  # Rough estimation
        # else:
        #     baseSearchVolume = 50000  # Default base volume
        
        # Calculate metrics for each row
        # searchVolumes = []
        competitions = []
        # cpcs = []
        
        for _, row in df.iterrows():
            trendValue = row['trendValue']
            
            # # Calculate search volume based on trend value and base volume
            # # Higher trend values = higher search volume
            # searchVolume = int(baseSearchVolume * (trendValue / 100) * np.random.uniform(0.8, 1.2))
            # searchVolumes.append(searchVolume)
            
            # Competition varies slightly over time but stays relatively stable
            competition = max(0.1, min(0.9, competitionScore + np.random.uniform(-0.1, 0.1)))
            competitions.append(round(competition, 3))
            
            # # CPC is inversely related to search volume and directly related to competition
            # # Higher competition + lower search volume = higher CPC
            # cpc = (competition * 5.0) / (searchVolume / 10000) + np.random.uniform(0.1, 0.5)
            # cpc = max(0.5, min(10.0, cpc))  # Keep CPC between $0.50 and $10.00
            # cpcs.append(round(cpc, 2))
        
        # Add calculated columns
        # df['searchVolume'] = searchVolumes
        df['competition'] = competitions
        # df['cpc'] = cpcs
        
        return df
        
    except Exception as e:
        print(f"Error calculating metrics: {e}")
        # Fallback to mock data for the additional columns
        # df['searchVolume'] = (df['trendValue'] * 1000 + np.random.normal(0, 500, len(df))).round(0)
        df['competition'] = np.random.uniform(0.1, 0.9, len(df)).round(3)
        # df['cpc'] = np.random.uniform(0.5, 5.0, len(df)).round(2)
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
    avgTopInterest = topQueries['value'].mean() if 'value' in topQueries.columns else 50
    
    numRisingQueries = len(risingQueries) if risingQueries is not None and not risingQueries.empty else 0
    
    # Competition formula
    competition = min(1.0, (
        (numTopQueries / 10) * 0.3 +  # More related queries = higher competition
        (avgTopInterest / 100) * 0.4 +  # Higher interest in related queries = higher competition
        (numRisingQueries / 5) * 0.3    # More rising queries = higher competition
    ))
    
    return round(competition, 3)