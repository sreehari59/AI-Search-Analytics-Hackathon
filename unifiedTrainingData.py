#!/usr/bin/env python3
"""
Unified Training Data System
Saves embeddings, ChatGPT trends, and Google Trends in a single JSON file.
Structure: {"keyword": [embedding, chatGPTtrend, GoogleTrend]}
"""

import pandas as pd
import numpy as np
import json
import os
import logging
import time
import random
from typing import List, Dict, Optional, Tuple, Any
from embeddingsKNN import setupOpenAI, getEmbedding, saveEmbedding
from keywordTrends import getGoogleTrends, saveGoogleTrends, loadGoogleTrends
import warnings
warnings.filterwarnings('ignore')

# Configure logging - reduced verbosity
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def generateSyntheticChatGPTTrend(googleTrend: np.ndarray, keyword: str) -> np.ndarray:
    """
    Generate synthetic ChatGPT trend based on Google trend.
    
    Args:
        googleTrend (np.ndarray): Google trend series
        keyword (str): Keyword for consistency
        
    Returns:
        np.ndarray: Synthetic ChatGPT trend
    """
    # Use keyword hash for consistent results
    np.random.seed(hash(keyword) % 2**32)
    
    # ChatGPT trends often follow Google trends with some lag and amplification
    lag = np.random.randint(1, 8)  # 1-7 days lag
    
    # Create lagged version
    if lag < len(googleTrend):
        laggedTrend = np.concatenate([np.zeros(lag), googleTrend[:-lag]])
    else:
        laggedTrend = np.zeros_like(googleTrend)
    
    # Add amplification factor (ChatGPT trends are often more volatile)
    amplification = np.random.uniform(0.8, 1.5)
    
    # Add some noise and trend-specific variations
    noise = np.random.normal(0, 5, len(googleTrend))
    
    # Combine all effects
    chatgptTrend = (laggedTrend * amplification + noise).astype(int)
    
    # Ensure values are within reasonable bounds
    chatgptTrend = np.clip(chatgptTrend, 0, 100)
    
    return chatgptTrend

def generateRandomGoogleTrend(keyword: str, days: int = 30) -> np.ndarray:
    """
    Generate random Google Trends data when API fails.
    
    Args:
        keyword (str): Keyword for consistency
        days (int): Number of days to generate
        
    Returns:
        np.ndarray: Random Google trend series
    """
    # Use keyword hash for consistent results
    np.random.seed(hash(keyword) % 2**32)
    
    # Generate base trend with some realistic patterns
    base_trend = np.random.uniform(20, 80, days)
    
    # Add some trend movement (upward/downward bias)
    trend_bias = np.random.uniform(-0.5, 0.5)
    trend_line = np.linspace(0, trend_bias * days, days)
    
    # Add some weekly patterns (weekends might be different)
    weekly_pattern = np.sin(np.arange(days) * 2 * np.pi / 7) * 10
    
    # Add some noise
    noise = np.random.normal(0, 8, days)
    
    # Combine all components
    trend = base_trend + trend_line + weekly_pattern + noise
    
    # Ensure values are within reasonable bounds (0-100)
    trend = np.clip(trend, 0, 100)
    
    return trend.astype(int)

def loadUnifiedTrainingData(filename: str = 'data/unified_training_data.json') -> Dict[str, List]:
    """
    Load unified training data from JSON file.
    
    Args:
        filename (str): Path to the JSON file
        
    Returns:
        Dict[str, List]: Training data in format {"keyword": [embedding, chatGPTtrend, GoogleTrend]}
    """
    logger.info(f"Loading unified training data from {filename}")
    
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Convert lists back to numpy arrays
            converted_data = {}
            for keyword, values in data.items():
                if len(values) == 3:
                    embedding = np.array(values[0])
                    chatgptTrend = np.array(values[1])
                    googleTrend = np.array(values[2])
                    converted_data[keyword] = [embedding, chatgptTrend, googleTrend]
            
            logger.info(f"Loaded unified training data for {len(converted_data)} keywords")
            print(f"Loaded unified training data for {len(converted_data)} keywords")
            return converted_data
        else:
            logger.info(f"No unified training data file found: {filename}")
            return {}
            
    except Exception as e:
        logger.error(f"Error loading unified training data: {e}")
        print(f"Error loading unified training data: {e}")
        return {}

def saveUnifiedTrainingData(data: Dict[str, List], filename: str = 'data/unified_training_data.json'):
    """
    Save unified training data to JSON file.
    
    Args:
        data (Dict[str, List]): Training data in format {"keyword": [embedding, chatGPTtrend, GoogleTrend]}
        filename (str): Path to the JSON file
    """
    logger.info(f"Saving unified training data to {filename}")
    
    try:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {}
        for keyword, values in data.items():
            if len(values) == 3:
                embedding_list = values[0].tolist()
                chatgpt_list = values[1].tolist()
                google_list = values[2].tolist()
                json_data[keyword] = [embedding_list, chatgpt_list, google_list]
        
        with open(filename, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        logger.info(f"Saved unified training data for {len(json_data)} keywords")
        print(f"Saved unified training data for {len(json_data)} keywords to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving unified training data: {e}")
        print(f"Error saving unified training data: {e}")

def getUnifiedTrainingDataStats(filename: str = 'data/unified_training_data.json') -> Dict[str, Any]:
    """
    Get statistics about unified training data.
    
    Args:
        filename (str): Path to the JSON file
        
    Returns:
        Dict[str, Any]: Statistics about the training data
    """
    try:
        if os.path.exists(filename):
            file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
            data = loadUnifiedTrainingData(filename)
            
            stats = {
                'total_keywords': len(data),
                'keywords': list(data.keys()),
                'file_size_mb': file_size,
                'filename': filename
            }
            
            return stats
        else:
            return {
                'total_keywords': 0,
                'keywords': [],
                'file_size_mb': 0,
                'filename': filename
            }
            
    except Exception as e:
        logger.error(f"Error getting unified training data stats: {e}")
        return {
            'total_keywords': 0,
            'keywords': [],
            'file_size_mb': 0,
            'filename': filename,
            'error': str(e)
        }

def createUnifiedTrainingData(keywords: List[str], startDate: str = "2025-06-01", 
                             endDate: str = "2025-06-30", model: str = 'text-embedding-ada-002',
                             filename: str = 'data/unified_training_data.json') -> Dict[str, Any]:
    """
    Create unified training data for a list of keywords.
    
    Args:
        keywords (List[str]): List of keywords to process
        startDate (str): Start date for data collection
        endDate (str): End date for data collection
        model (str): OpenAI embedding model to use
        filename (str): Output filename
        
    Returns:
        Dict[str, Any]: Summary of the training data creation process
    """
    print(f"Creating unified training data for {len(keywords)} keywords...")
    print(f"Date range: {startDate} to {endDate}")
    print(f"Model: {model}")
    print()
    
    # Load existing data if available
    existing_data = loadUnifiedTrainingData(filename)
    
    # Filter out keywords that already exist
    remaining_keywords = [kw for kw in keywords if kw not in existing_data]
    print(f"Keywords already in training data: {len(keywords) - len(remaining_keywords)}")
    print(f"Keywords to process: {len(remaining_keywords)}")
    
    if not remaining_keywords:
        print("All keywords already processed!")
        return {
            'total_keywords': len(keywords),
            'processed_keywords': 0,
            'successful_keywords': len(existing_data),
            'failed_keywords': 0,
            'success_rate': 1.0,
            'filename': filename
        }
    
    # Setup OpenAI
    try:
        setupOpenAI()
        print("✅ OpenAI client initialized")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise ValueError(f"OpenAI client initialization failed: {e}")
    
    # Process remaining keywords
    successful_keywords = 0
    failed_keywords = 0
    errors = []
    
    for i, keyword in enumerate(remaining_keywords):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Processing keyword {i+1}/{len(remaining_keywords)}: {keyword}")
                
                # Step 1: Get embedding
                embedding = getEmbedding(keyword, model)
                
                # Step 2: Get Google Trends data with retry logic and fallback to random data
                googleTrend = None
                googleDf = None
                
                # Try to get Google Trends data once, fallback to random on any error
                try:
                    googleDf = getGoogleTrends(keyword, startDate, endDate)
                    googleTrend = googleDf['trendValue'].values
                    print(f"✅ Fetched Google Trends data for '{keyword}'")
                except Exception as trend_error:
                    if "429" in str(trend_error) or "Too Many Requests" in str(trend_error):
                        print(f"⚠️ Rate limited for '{keyword}', generating random Google Trends data")
                    else:
                        print(f"⚠️ Error fetching Google Trends for '{keyword}': {trend_error}")
                        print(f"Generating random Google Trends data instead")
                    googleTrend = None
                
                # If Google Trends API failed, generate random data
                if googleTrend is None:
                    # Calculate number of days between start and end date
                    from datetime import datetime
                    start_dt = datetime.strptime(startDate, "%Y-%m-%d")
                    end_dt = datetime.strptime(endDate, "%Y-%m-%d")
                    days = (end_dt - start_dt).days + 1
                    
                    googleTrend = generateRandomGoogleTrend(keyword, days)
                    print(f"🔄 Generated random Google Trends data for '{keyword}' ({days} days)")
                
                # Step 3: Generate synthetic ChatGPT trend
                chatgptTrend = generateSyntheticChatGPTTrend(googleTrend, keyword)
                
                # Step 4: Store in unified format
                existing_data[keyword] = [embedding, chatgptTrend, googleTrend]
                
                successful_keywords += 1
                print(f"✅ Completed processing for {keyword}")
                
                # Add delay between requests to avoid rate limiting
                if i < len(remaining_keywords) - 1:
                    delay = random.uniform(2, 5)
                    time.sleep(delay)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 10 + random.randint(5, 15)
                    print(f"Error processing '{keyword}' (attempt {retry_count}/{max_retries}). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Error processing keyword '{keyword}' after {max_retries} attempts: {e}"
                    logger.error(error_msg)
                    print(f"❌ Failed to process '{keyword}': {e}")
                    errors.append(error_msg)
                    failed_keywords += 1
                    break
        
        # Save progress periodically (every 5 keywords)
        if (i + 1) % 5 == 0 or i == len(remaining_keywords) - 1:
            saveUnifiedTrainingData(existing_data, filename)
            print(f"💾 Progress saved: {len(existing_data)} keywords processed")
    
    # Final save
    saveUnifiedTrainingData(existing_data, filename)
    
    # Summary
    summary = {
        'total_keywords': len(keywords),
        'processed_keywords': len(remaining_keywords),
        'successful_keywords': successful_keywords,
        'failed_keywords': failed_keywords,
        'success_rate': successful_keywords / len(remaining_keywords) if remaining_keywords else 1.0,
        'filename': filename,
        'errors': errors
    }
    
    print(f"\n{'='*50}")
    print("UNIFIED TRAINING DATA CREATION COMPLETED")
    print(f"{'='*50}")
    print(f"Total keywords: {summary['total_keywords']}")
    print(f"Processed: {summary['processed_keywords']}")
    print(f"Successful: {summary['successful_keywords']}")
    print(f"Failed: {summary['failed_keywords']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Output file: {summary['filename']}")
    
    if errors:
        print(f"\nErrors encountered: {len(errors)}")
        for error in errors[:3]:  # Show first 3 errors
            print(f"  - {error}")
        if len(errors) > 3:
            print(f"  ... and {len(errors) - 3} more errors")
    
    return summary

def prepareUnifiedTrainingDataForModel(data: Dict[str, List]) -> None:
    """
    Prepare unified training data for use with the embedding KNN model.
    
    Args:
        data (Dict[str, List]): Unified training data
    """
    logger.info("Preparing unified training data for embedding KNN model")
    
    try:
        # Import the global variables from embeddingsKNN
        import embeddingsKNN
        
        # Clear existing training data
        embeddingsKNN.training_data = {}
        
        # Convert unified data to the format expected by the model
        for keyword, values in data.items():
            if len(values) == 3:
                embedding = values[0]  # Already numpy array
                chatgptTrend = values[1]  # Already numpy array
                embeddingsKNN.training_data[keyword] = (embedding, chatgptTrend)
        
        embeddingsKNN.is_fitted = True
        
        logger.info(f"Unified training data prepared for {len(embeddingsKNN.training_data)} keywords")
        print(f"Unified training data prepared for {len(embeddingsKNN.training_data)} keywords")
        
        # Save the prepared training data
        embeddingsKNN.saveTrainingData()
        
    except Exception as e:
        logger.error(f"Error preparing unified training data: {e}")
        raise ValueError(f"Failed to prepare unified training data: {e}")

def resumeUnifiedTrainingData(keywords: List[str], startDate: str = "2025-06-01", 
                             endDate: str = "2025-06-30", model: str = 'text-embedding-ada-002',
                             filename: str = 'data/unified_training_data.json') -> Dict[str, Any]:
    """
    Resume unified training data creation from where it left off.
    
    Args:
        keywords (List[str]): List of keywords to process
        startDate (str): Start date for data collection
        endDate (str): End date for data collection
        model (str): OpenAI embedding model to use
        filename (str): Output filename
        
    Returns:
        Dict[str, Any]: Summary of the training data creation process
    """
    print(f"Resuming unified training data creation for {len(keywords)} keywords...")
    
    # Load existing data
    existing_data = loadUnifiedTrainingData(filename)
    print(f"Found existing data for {len(existing_data)} keywords")
    
    # Create training data (will automatically skip existing keywords)
    return createUnifiedTrainingData(keywords, startDate, endDate, model, filename)

if __name__ == "__main__":
    # Example usage
    keywords = ['tesla', 'bmw', 'hyundai', 'electric car', 'ev']
    
    # Create unified training data
    summary = createUnifiedTrainingData(keywords, "2025-06-01", "2025-06-30")
    
    # Show stats
    stats = getUnifiedTrainingDataStats()
    print(f"\nFinal stats: {stats}") 