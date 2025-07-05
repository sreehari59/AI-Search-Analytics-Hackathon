import pandas as pd
import numpy as np
import json
import os
import logging
import time
import random
from typing import List, Dict, Optional
from embeddingsKNN import setupOpenAI, getEmbedding, saveEmbedding, fitModel, saveTrainingData
from keywordTrends import getGoogleTrends
import warnings
warnings.filterwarnings('ignore')

# Configure logging - reduced verbosity
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_data_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def generateSyntheticChatGPTTrend(googleTrend: np.ndarray, keyword: str) -> np.ndarray:
    """
    Generate synthetic ChatGPT trend based on Google trend with realistic patterns.
    
    Args:
        googleTrend (np.ndarray): Google trend series
        keyword (str): Keyword for consistency
        
    Returns:
        np.ndarray: Synthetic ChatGPT trend
    """
    # Use keyword hash for consistent results
    np.random.seed(hash(keyword) % 2**32)
    
    # ChatGPT trends often follow Google trends with some lag and amplification
    # Add lag effect (ChatGPT trends often follow Google trends)
    lag = np.random.randint(1, 8)  # 1-7 days lag
    laggedTrend = np.roll(googleTrend, lag)
    
    # Add amplification factor (ChatGPT trends can be more volatile)
    amplification = np.random.uniform(0.8, 1.5)
    
    # Add noise
    noise = np.random.normal(0, 5, len(googleTrend))
    
    # Combine effects
    chatgptTrend = laggedTrend * amplification + noise
    
    # Ensure values are reasonable (0-100 range)
    chatgptTrend = np.clip(chatgptTrend, 0, 100)
    
    return chatgptTrend


def generateTrainingData(keywords: List[str], startDate: str = "2023-01-01", endDate: str = "2023-12-31", 
                        model: str = 'text-embedding-ada-002', saveEmbeddings: bool = True) -> Dict[str, any]:
    """
    Generate training data for the embedding KNN model.
    
    Args:
        keywords (List[str]): List of keywords to generate training data for
        startDate (str): Start date for data collection
        endDate (str): End date for data collection
        model (str): OpenAI embedding model to use
        saveEmbeddings (bool): Whether to save embeddings to file
        
    Returns:
        Dict[str, any]: Training data generation results
    """
    logger.info(f"Starting training data generation for {len(keywords)} keywords")
    logger.info(f"Date range: {startDate} to {endDate}")
    logger.info(f"Using model: {model}")
    
    # Check and initialize OpenAI client if needed
    try:
        setupOpenAI()
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        raise ValueError(f"OpenAI client initialization failed: {e}")
    
    trainingData = {}
    successfulKeywords = 0
    failedKeywords = 0
    errors = []
    
    for i, keyword in enumerate(keywords):
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                print(f"Creating training data for {keyword}")
                
                # Step 1: Get embedding for the keyword
                embedding = getEmbedding(keyword, model)
                
                # Step 2: Get Google Trends data with retry logic
                googleDf = None
                for trend_retry in range(max_retries):
                    try:
                        googleDf = getGoogleTrends(keyword, startDate, endDate)
                        break
                    except Exception as trend_error:
                        if "429" in str(trend_error) or "Too Many Requests" in str(trend_error):
                            wait_time = (trend_retry + 1) * 30 + random.randint(5, 15)  # 30s, 60s, 90s + random
                            print(f"Rate limited for {keyword}. Waiting {wait_time} seconds...")
                            time.sleep(wait_time)
                            continue
                        else:
                            raise trend_error
                
                if googleDf is None:
                    raise Exception("Failed to fetch Google Trends data after all retries")
                
                googleTrend = googleDf['trendValue'].values
                
                # Step 3: Generate synthetic ChatGPT trend
                chatgptTrend = generateSyntheticChatGPTTrend(googleTrend, keyword)
                
                # Step 4: Store the data
                trainingData[keyword] = {
                    'embedding': embedding.tolist(),  # Convert to list for JSON serialization
                    'google_trend': googleTrend.tolist(),
                    'chatgpt_trend': chatgptTrend.tolist(),
                    'google_stats': {
                        'mean': float(np.mean(googleTrend)),
                        'max': float(np.max(googleTrend)),
                        'min': float(np.min(googleTrend)),
                        'std': float(np.std(googleTrend))
                    },
                    'chatgpt_stats': {
                        'mean': float(np.mean(chatgptTrend)),
                        'max': float(np.max(chatgptTrend)),
                        'min': float(np.min(chatgptTrend)),
                        'std': float(np.std(chatgptTrend))
                    }
                }
                
                successfulKeywords += 1
                print(f"Created training data for {keyword}")
                
                # Add delay between requests to avoid rate limiting
                if i < len(keywords) - 1:  # Don't wait after the last keyword
                    delay = random.uniform(2, 5)  # Random delay between 2-5 seconds
                    time.sleep(delay)
                
                break  # Success, exit retry loop
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = retry_count * 10 + random.randint(5, 15)  # 10s, 20s, 30s + random
                    print(f"Error processing '{keyword}' (attempt {retry_count}/{max_retries}). Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Error processing keyword '{keyword}' after {max_retries} attempts: {e}"
                    logger.error(error_msg)
                    print(f"Error processing keyword '{keyword}': {e}")
                    errors.append(error_msg)
                    failedKeywords += 1
                    break
    
    # Save training data
    trainingDataFile = f"data/training_data_{startDate.replace('-', '')}_{endDate.replace('-', '')}.json"
    
    try:
        with open(trainingDataFile, 'w') as f:
            json.dump(trainingData, f, indent=2)
        print(f"Training data saved to {trainingDataFile}")
    except Exception as e:
        logger.error(f"Error saving training data: {e}")
        print(f"Error saving training data: {e}")
    
    # Generate summary
    summary = {
        'total_keywords': len(keywords),
        'successful_keywords': successfulKeywords,
        'failed_keywords': failedKeywords,
        'success_rate': successfulKeywords / len(keywords) if keywords else 0,
        'training_data_file': trainingDataFile,
        'date_range': f"{startDate} to {endDate}",
        'model_used': model,
        'errors': errors
    }
    
    print(f"\nTraining Data Generation Summary:")
    print(f"Total keywords: {summary['total_keywords']}")
    print(f"Successful: {summary['successful_keywords']}")
    print(f"Failed: {summary['failed_keywords']}")
    print(f"Success rate: {summary['success_rate']:.2%}")
    print(f"Training data saved to: {summary['training_data_file']}")
    
    return summary


def loadTrainingData(fileName: str) -> Dict[str, any]:
    """
    Load training data from JSON file.
    
    Args:
        fileName (str): Path to training data file
        
    Returns:
        Dict[str, any]: Training data
    """
    logger.info(f"Loading training data from {fileName}")
    
    try:
        if os.path.exists(fileName):
            with open(fileName, 'r') as f:
                trainingData = json.load(f)
            logger.info(f"Training data loaded successfully from {fileName}")
            logger.info(f"Loaded data for {len(trainingData)} keywords")
            return trainingData
        else:
            logger.warning(f"Training data file {fileName} not found")
            return {}
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        return {}


def getTrainingDataStats(fileName: str) -> Dict[str, any]:
    """
    Get statistics about training data.
    
    Args:
        fileName (str): Path to training data file
        
    Returns:
        Dict[str, any]: Training data statistics
    """
    logger.info(f"Getting training data statistics from {fileName}")
    
    try:
        trainingData = loadTrainingData(fileName)
        
        if not trainingData:
            return {'total_keywords': 0, 'file_size_mb': 0, 'keywords': []}
        
        # Calculate statistics
        googleMeans = [data['google_stats']['mean'] for data in trainingData.values()]
        chatgptMeans = [data['chatgpt_stats']['mean'] for data in trainingData.values()]
        
        stats = {
            'total_keywords': len(trainingData),
            'file_size_mb': os.path.getsize(fileName) / (1024 * 1024) if os.path.exists(fileName) else 0,
            'keywords': list(trainingData.keys()),
            'google_trends': {
                'mean': float(np.mean(googleMeans)),
                'std': float(np.std(googleMeans)),
                'min': float(np.min(googleMeans)),
                'max': float(np.max(googleMeans))
            },
            'chatgpt_trends': {
                'mean': float(np.mean(chatgptMeans)),
                'std': float(np.std(chatgptMeans)),
                'min': float(np.min(chatgptMeans)),
                'max': float(np.max(chatgptMeans))
            }
        }
        
        logger.info(f"Training data stats: {stats['total_keywords']} keywords, {stats['file_size_mb']:.2f} MB")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting training data stats: {e}")
        return {'total_keywords': 0, 'file_size_mb': 0, 'keywords': [], 'error': str(e)}


def prepareTrainingDataForModel(trainingData: Dict[str, any]) -> None:
    """
    Prepare training data for use with the embedding KNN model.
    This converts the JSON format to the format expected by fitModel.
    
    Args:
        trainingData (Dict[str, any]): Training data from JSON file
    """
    logger.info("Preparing training data for embedding KNN model")
    
    try:
        # Import the global variables from embeddingsKNN
        import embeddingsKNN
        
        # Clear existing training data
        embeddingsKNN.training_data = {}
        
        # Convert training data to the format expected by the model
        for keyword, data in trainingData.items():
            embedding = np.array(data['embedding'])
            chatgptTrend = np.array(data['chatgpt_trend'])
            embeddingsKNN.training_data[keyword] = (embedding, chatgptTrend)
        
        embeddingsKNN.is_fitted = True
        
        logger.info(f"Training data prepared for {len(embeddingsKNN.training_data)} keywords")
        print(f"Training data prepared for {len(embeddingsKNN.training_data)} keywords")
        
        # Save the prepared training data
        embeddingsKNN.saveTrainingData()
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        raise ValueError(f"Failed to prepare training data: {e}")


def resumeTrainingDataGeneration(keywords: List[str], startDate: str = "2025-06-01", endDate: str = "2025-06-30", 
                                model: str = 'text-embedding-ada-002', existingFile: str = None) -> Dict[str, any]:
    """
    Resume training data generation from where it left off.
    
    Args:
        keywords (List[str]): List of keywords to generate training data for
        startDate (str): Start date for data collection
        endDate (str): End date for data collection
        model (str): OpenAI embedding model to use
        existingFile (str): Path to existing training data file to resume from
        
    Returns:
        Dict[str, any]: Training data generation results
    """
    print(f"Resuming training data generation for {len(keywords)} keywords...")
    
    # Load existing data if provided
    existingData = {}
    if existingFile and os.path.exists(existingFile):
        try:
            with open(existingFile, 'r') as f:
                existingData = json.load(f)
            print(f"Loaded existing data for {len(existingData)} keywords from {existingFile}")
        except Exception as e:
            print(f"Error loading existing data: {e}")
    
    # Filter out keywords that already exist
    remainingKeywords = [kw for kw in keywords if kw not in existingData]
    print(f"Remaining keywords to process: {len(remainingKeywords)}")
    
    if not remainingKeywords:
        print("All keywords already processed!")
        return {
            'total_keywords': len(keywords),
            'successful_keywords': len(existingData),
            'failed_keywords': 0,
            'success_rate': 1.0,
            'training_data_file': existingFile,
            'date_range': f"{startDate} to {endDate}",
            'model_used': model,
            'errors': []
        }
    
    # Generate data for remaining keywords
    summary = generateTrainingData(
        keywords=remainingKeywords,
        startDate=startDate,
        endDate=endDate,
        model=model,
        saveEmbeddings=True
    )
    
    # Merge with existing data
    if existingData:
        mergedData = {**existingData, **summary.get('training_data', {})}
        
        # Save merged data
        mergedFile = f"data/training_data_{startDate.replace('-', '')}_{endDate.replace('-', '')}_merged.json"
        try:
            with open(mergedFile, 'w') as f:
                json.dump(mergedData, f, indent=2)
            print(f"Merged training data saved to {mergedFile}")
            
            # Update summary
            summary['training_data_file'] = mergedFile
            summary['total_keywords'] = len(keywords)
            summary['successful_keywords'] = len(mergedData)
            summary['success_rate'] = len(mergedData) / len(keywords)
            
        except Exception as e:
            print(f"Error saving merged data: {e}")
    
    return summary