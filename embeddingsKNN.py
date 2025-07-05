import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Dict
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from keywordTrends import getGoogleTrends
from openai import OpenAI
import os
import json
import pickle
import logging
import time
import random
import warnings
warnings.filterwarnings('ignore')

# Configure logging - reduced verbosity
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('embeddings_knn.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables to store data
training_data = {}  # Store keyword embeddings and ChatGPT trends
keyword_embeddings = {}  # Cache for embeddings
is_fitted = False
openai_client = None  # OpenAI client instance

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

def migrateEmbeddingsIfNeeded():
    """Migrate embeddings from old filename to new filename if needed."""
    old_filename = "data/keyWordsEmbeddings.json"
    new_filename = "data/keywordsEmbeddings.json"
    
    if os.path.exists(old_filename) and not os.path.exists(new_filename):
        logger.info("Migrating embeddings from old filename to new filename")
        try:
            import shutil
            shutil.move(old_filename, new_filename)
            logger.info(f"Successfully migrated {old_filename} to {new_filename}")
        except Exception as e:
            logger.error(f"Error migrating embeddings: {e}")

# Run migration on module import
migrateEmbeddingsIfNeeded()

def setupOpenAI(apiKey: Optional[str] = None):
    """
    Setup OpenAI API key.
    
    Args:
        apiKey (str, optional): OpenAI API key
    """
    global openai_client
    logger.info("Setting up OpenAI API configuration")
    
    if apiKey:
        openai_client = OpenAI(api_key=apiKey)
        logger.info("OpenAI API key provided directly")
    elif 'OPENAI_API_KEY' in os.environ:
        openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
        logger.info("OpenAI API key loaded from environment variable")
    else:
        logger.error("No OpenAI API key found")
        raise ValueError("OpenAI API key must be provided or set in OPENAI_API_KEY environment variable")
    
    logger.info("OpenAI API setup completed successfully")


def saveEmbedding(keyword: str, embedding: np.ndarray, model: str = 'text-embedding-ada-002'):
    """
    Save embedding to a single JSON file.
    
    Args:
        keyword (str): Keyword
        embedding (np.ndarray): Embedding vector
        model (str): Model name
    """
    logger.info(f"Saving embedding for keyword: '{keyword}' with model: {model}")
    
    try:
        # Create filename for the single embeddings file
        filename = f"data/keywordsEmbeddings.json"
        
        # Load existing embeddings if file exists
        embeddings_data = {}
        if os.path.exists(filename):
            try:
                with open(filename, 'r') as f:
                    embeddings_data = json.load(f)
                logger.info(f"Loaded existing embeddings from {filename}")
            except Exception as e:
                logger.warning(f"Error loading existing embeddings, starting fresh: {e}")
                embeddings_data = {}
        
        # Add new embedding (convert numpy array to list for JSON serialization)
        embeddings_data[keyword] = embedding.tolist()
        
        # Save all embeddings back to file
        with open(filename, 'w') as f:
            json.dump(embeddings_data, f, indent=2)
        
        logger.info(f"Embedding saved successfully for '{keyword}' to {filename}")
        print(f"Embedding saved for '{keyword}' to {filename}")
        
    except Exception as e:
        logger.error(f"Error saving embedding for '{keyword}': {e}")
        print(f"Error saving embedding for '{keyword}': {e}")


def loadEmbedding(keyword: str, model: str = 'text-embedding-ada-002') -> Optional[np.ndarray]:
    """
    Load embedding from the single JSON file.
    
    Args:
        keyword (str): Keyword
        model (str): Model name
        
    Returns:
        Optional[np.ndarray]: Embedding vector if found, None otherwise
    """
    logger.info(f"Attempting to load embedding for keyword: '{keyword}' with model: {model}")
    
    try:
        # Create filename for the single embeddings file
        filename = f"data/keywordsEmbeddings.json"
        
        # Check if file exists
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                embeddings_data = json.load(f)
            
            # Check if keyword exists in the data
            if keyword in embeddings_data:
                # Convert list back to numpy array
                embedding = np.array(embeddings_data[keyword])
                logger.info(f"Embedding loaded successfully for '{keyword}' from {filename}")
                print(f"Embedding loaded for '{keyword}' from {filename}")
                return embedding
            else:
                logger.info(f"Keyword '{keyword}' not found in embeddings file")
                return None
        
        logger.info(f"No embeddings file found: {filename}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading embedding for '{keyword}': {e}")
        print(f"Error loading embedding for '{keyword}': {e}")
        return None


def saveTrainingData(fileName: str = 'data/training_data.pkl'):
    """
    Save training data to file.
    
    Args:
        fileName (str): Filename to save to
    """
    global training_data
    logger.info(f"Saving training data to {fileName}")
    
    try:
        with open(fileName, 'wb') as f:
            pickle.dump(training_data, f)
        logger.info(f"Training data saved successfully to {fileName} with {len(training_data)} keywords")
        print(f"Training data saved to {fileName}")
    except Exception as e:
        logger.error(f"Error saving training data: {e}")
        print(f"Error saving training data: {e}")


def loadTrainingData(fileName: str = 'data/training_data.pkl'):
    """
    Load training data from file.
    
    Args:
        fileName (str): Filename to load from
    """
    global training_data, is_fitted
    logger.info(f"Attempting to load training data from {fileName}")
    
    try:
        if os.path.exists(fileName):
            with open(fileName, 'rb') as f:
                training_data = pickle.load(f)
            is_fitted = True
            logger.info(f"Training data loaded successfully from {fileName} with {len(training_data)} keywords")
            print(f"Training data loaded from {fileName}")
            print(f"Loaded data for {len(training_data)} keywords")
        else:
            logger.warning(f"Training data file {fileName} not found")
            print(f"Training data file {fileName} not found")
    except Exception as e:
        logger.error(f"Error loading training data: {e}")
        print(f"Error loading training data: {e}")


def getAllSavedEmbeddings(model: str = 'text-embedding-ada-002') -> Dict[str, np.ndarray]:
    """
    Get all saved embeddings from the JSON file.
    
    Args:
        model (str): Model name
        
    Returns:
        Dict[str, np.ndarray]: Dictionary of keyword to embedding mapping
    """
    logger.info(f"Loading all saved embeddings for model: {model}")
    
    try:
        filename = f"data/keywordsEmbeddings.json"
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                embeddings_data = json.load(f)
            
            # Convert lists back to numpy arrays
            embeddings = {keyword: np.array(embedding_list) for keyword, embedding_list in embeddings_data.items()}
            
            logger.info(f"Loaded {len(embeddings)} embeddings from {filename}")
            return embeddings
        else:
            logger.info(f"No embeddings file found: {filename}")
            return {}
            
    except Exception as e:
        logger.error(f"Error loading all embeddings: {e}")
        return {}


def getEmbeddingsStats(model: str = 'text-embedding-ada-002') -> Dict[str, any]:
    """
    Get statistics about saved embeddings.
    
    Args:
        model (str): Model name
        
    Returns:
        Dict[str, any]: Statistics about the embeddings
    """
    logger.info(f"Getting embeddings statistics for model: {model}")
    
    try:
        filename = f"data/keywordsEmbeddings.json"
        
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                embeddings_data = json.load(f)
            
            stats = {
                'total_embeddings': len(embeddings_data),
                'keywords': list(embeddings_data.keys()),
                'file_size_mb': os.path.getsize(filename) / (1024 * 1024),
                'model': model
            }
            
            logger.info(f"Embeddings stats: {stats['total_embeddings']} embeddings, {stats['file_size_mb']:.2f} MB")
            return stats
        else:
            logger.info(f"No embeddings file found: {filename}")
            return {'total_embeddings': 0, 'keywords': [], 'file_size_mb': 0, 'model': model}
            
    except Exception as e:
        logger.error(f"Error getting embeddings stats: {e}")
        return {'total_embeddings': 0, 'keywords': [], 'file_size_mb': 0, 'model': model, 'error': str(e)}


def analyzeNewKeyword(keyword: str, startDate: Optional[str] = None, endDate: Optional[str] = None, k: int = 3, weights: str = 'distance', model: str = 'text-embedding-ada-002') -> Dict[str, any]:
    """
    Complete analysis of a new keyword: get Google trends, predict ChatGPT trends, and provide insights.
    
    Args:
        keyword (str): New keyword to analyze
        startDate (str, optional): Start date for data collection (YYYY-MM-DD)
        endDate (str, optional): End date for data collection (YYYY-MM-DD)
        k (int): Number of nearest neighbors to consider for prediction
        weights (str): Weighting scheme ('uniform' or 'distance')
        model (str): OpenAI embedding model to use
        
    Returns:
        Dict[str, any]: Complete analysis results including:
            - google_trends: Google Trends data
            - predicted_chatgpt_trend: Predicted ChatGPT trend
            - neighbors: Nearest neighbor keywords and similarities
            - analysis_summary: Summary statistics and insights
    """
    print(f"Running inference for {keyword}")
    
    # Check and initialize OpenAI client if needed
    global openai_client
    if openai_client is None:
        try:
            setupOpenAI()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"OpenAI client initialization failed: {e}")
    
    try:
        # Step 1: Get Google Trends data with immediate fallback to random data
        googleTrend = None
        googleDf = None
        
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
            start_dt = datetime.strptime(startDate, "%Y-%m-%d") if startDate else datetime.now()
            end_dt = datetime.strptime(endDate, "%Y-%m-%d") if endDate else datetime.now()
            days = (end_dt - start_dt).days + 1
            
            # Generate random Google Trends data
            np.random.seed(hash(keyword) % 2**32)
            base_trend = np.random.uniform(20, 80, days)
            trend_bias = np.random.uniform(-0.5, 0.5)
            trend_line = np.linspace(0, trend_bias * days, days)
            weekly_pattern = np.sin(np.arange(days) * 2 * np.pi / 7) * 10
            noise = np.random.normal(0, 8, days)
            googleTrend = np.clip(base_trend + trend_line + weekly_pattern + noise, 0, 100).astype(int)
            print(f"🔄 Generated random Google Trends data for '{keyword}' ({days} days)")
            
            # Create a proper DataFrame structure for the analysis
            googleDf = pd.DataFrame({
                'date': pd.date_range(start=startDate or '2025-06-01', periods=days, freq='D'),
                'trendValue': googleTrend,
                'isPartial': [False] * days,
                'keyword': [keyword] * days
            })
        
        # Step 2: Get embedding for the keyword
        embedding = getEmbedding(keyword, model)
        
        # Step 3: Predict ChatGPT trend using KNN
        predictedChatgptTrend, neighborsInfo = predictTrendGPT(keyword, k, weights, startDate, endDate)
        
        # Step 4: Generate synthetic actual ChatGPT trend for comparison
        actualChatgptTrend = generateSyntheticsTrendGPT(googleTrend, keyword)
        
        # Step 5: Calculate analysis metrics
        
        # Google Trends statistics
        googleStats = {
            'mean_trend': float(np.mean(googleTrend)),
            'max_trend': float(np.max(googleTrend)),
            'min_trend': float(np.min(googleTrend)),
            'trend_volatility': float(np.std(googleTrend)),
            'trend_range': float(np.max(googleTrend) - np.min(googleTrend))
        }
        
        # Predicted ChatGPT statistics
        predictedStats = {
            'mean_predicted': float(np.mean(predictedChatgptTrend)),
            'max_predicted': float(np.max(predictedChatgptTrend)),
            'min_predicted': float(np.min(predictedChatgptTrend)),
            'predicted_volatility': float(np.std(predictedChatgptTrend))
        }
        
        # Actual ChatGPT statistics
        actualStats = {
            'mean_actual': float(np.mean(actualChatgptTrend)),
            'max_actual': float(np.max(actualChatgptTrend)),
            'min_actual': float(np.min(actualChatgptTrend)),
            'actual_volatility': float(np.std(actualChatgptTrend))
        }
        
        # Prediction accuracy metrics
        mse = mean_squared_error(actualChatgptTrend, predictedChatgptTrend)
        mae = mean_absolute_error(actualChatgptTrend, predictedChatgptTrend)
        
        # Neighbor information
        neighbors = [
            {
                'keyword': neighbor[0],
                'similarity': float(1 - neighbor[1]),  # Convert distance to similarity
                'distance': float(neighbor[1])
            }
            for neighbor in neighborsInfo
        ]
        
        # Create comprehensive analysis result
        analysis_result = {
            'keyword': keyword,
            'google_trends': {
                'data': googleDf.to_dict('records'),
                'trend_values': googleTrend.tolist(),
                'statistics': googleStats
            },
            'predicted_chatgpt_trend': {
                'trend_values': predictedChatgptTrend.tolist(),
                'statistics': predictedStats
            },
            'actual_chatgpt_trend': {
                'trend_values': actualChatgptTrend.tolist(),
                'statistics': actualStats
            },
            'neighbors': neighbors,
            'prediction_metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse))
            },
            'analysis_summary': {
                'trend_correlation': float(np.corrcoef(googleTrend, predictedChatgptTrend)[0, 1]),
                'prediction_confidence': float(1 - np.mean([n['distance'] for n in neighbors])),
                'trend_magnitude': 'high' if googleStats['mean_trend'] > 50 else 'medium' if googleStats['mean_trend'] > 25 else 'low',
                'volatility_level': 'high' if googleStats['trend_volatility'] > 20 else 'medium' if googleStats['trend_volatility'] > 10 else 'low'
            },
            'metadata': {
                'analysis_date': pd.Timestamp.now().isoformat(),
                'model_used': model,
                'k_neighbors': k,
                'weighting_scheme': weights,
                'date_range': f"{startDate} to {endDate}" if startDate and endDate else "default"
            }
        }
        
        print(f"Completed inference for {keyword}")
        return analysis_result
        
    except Exception as e:
        logger.error(f"Error analyzing keyword '{keyword}': {e}")
        raise ValueError(f"Failed to analyze keyword '{keyword}': {e}")


def quickKeywordAnalysis(keyword: str, startDate: Optional[str] = None, endDate: Optional[str] = None) -> Dict[str, any]:
    """
    Quick analysis of a keyword with default settings.
    
    Args:
        keyword (str): Keyword to analyze
        startDate (str, optional): Start date for data collection
        endDate (str, optional): End date for data collection
        
    Returns:
        Dict[str, any]: Quick analysis results
    """
    print(f"Running inference for {keyword}")
    
    # Check and initialize OpenAI client if needed
    global openai_client
    if openai_client is None:
        try:
            setupOpenAI()
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise ValueError(f"OpenAI client initialization failed: {e}")
    
    try:
        # Use default settings for quick analysis with immediate fallback
        result = None
        
        try:
            result = analyzeNewKeyword(
                keyword=keyword,
                startDate=startDate,
                endDate=endDate,
                k=3,
                weights='distance',
                model='text-embedding-ada-002'
            )
        except Exception as analysis_error:
            if "429" in str(analysis_error) or "Too Many Requests" in str(analysis_error):
                print(f"⚠️ Rate limited for '{keyword}', but analysis will continue with fallback data")
                # The analyzeNewKeyword function now handles rate limits internally
                # so we can retry once more
                try:
                    result = analyzeNewKeyword(
                        keyword=keyword,
                        startDate=startDate,
                        endDate=endDate,
                        k=3,
                        weights='distance',
                        model='text-embedding-ada-002'
                    )
                except Exception as retry_error:
                    raise retry_error
            else:
                raise analysis_error
        
        if result is None:
            raise Exception("Failed to analyze keyword")
        
        # Return simplified version for quick analysis
        quick_result = {
            'keyword': keyword,
            'google_trend_mean': result['google_trends']['statistics']['mean_trend'],
            'predicted_chatgpt_mean': result['predicted_chatgpt_trend']['statistics']['mean_predicted'],
            'prediction_confidence': result['analysis_summary']['prediction_confidence'],
            'nearest_neighbors': [n['keyword'] for n in result['neighbors']],
            'trend_magnitude': result['analysis_summary']['trend_magnitude'],
            'volatility_level': result['analysis_summary']['volatility_level']
        }
        
        print(f"Completed inference for {keyword}")
        return quick_result
        
    except Exception as e:
        logger.error(f"Error in quick analysis for '{keyword}': {e}")
        raise ValueError(f"Quick analysis failed for '{keyword}': {e}")


def getEmbedding(keyword: str, model: str = 'text-embedding-ada-002') -> np.ndarray:
    """
    Get OpenAI embedding for a keyword.
    
    Args:
        keyword (str): Keyword to embed
        model (str): OpenAI embedding model to use
        
    Returns:
        np.ndarray: Embedding vector
    """
    global keyword_embeddings
    
    logger.info(f"Getting embedding for keyword: '{keyword}' with model: {model}")
    
    # Check cache first
    if keyword in keyword_embeddings:
        logger.info(f"Embedding found in cache for '{keyword}'")
        print(f"Using cached embedding for '{keyword}'")
        return keyword_embeddings[keyword]
    
    # Check if embedding exists in saved file
    embedding = loadEmbedding(keyword, model)
    if embedding is not None:
        keyword_embeddings[keyword] = embedding
        logger.info(f"Embedding loaded from file and cached for '{keyword}'")
        print(f"Loaded existing embedding for '{keyword}' from file")
        return embedding
    
    logger.info(f"Fetching embedding from OpenAI API for '{keyword}'")
    print(f"Creating new embedding for '{keyword}' via OpenAI API")
    
    if openai_client is None:
        logger.error("OpenAI client not initialized. Call setupOpenAI() first.")
        raise ValueError("OpenAI client not initialized. Call setupOpenAI() first.")
    
    try:
        # Get embedding from OpenAI using new client API
        response = openai_client.embeddings.create(
            input=keyword,
            model=model
        )
        
        embedding = np.array(response.data[0].embedding)
        
        # Cache the embedding
        keyword_embeddings[keyword] = embedding
        
        # Save the embedding
        saveEmbedding(keyword, embedding, model)
        
        logger.info(f"Embedding successfully fetched and cached for '{keyword}'")
        return embedding
        
    except Exception as e:
        logger.error(f"Error getting embedding for '{keyword}': {e}")
        print(f"Error getting embedding for '{keyword}': {e}")
        # Return zero vector as fallback
        logger.warning(f"Returning zero vector as fallback for '{keyword}'")
        return np.zeros(1536)  # Default embedding size for text-embedding-ada-002


def cosineSimilarityDistance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity distance between two embeddings.
    
    Args:
        embedding1 (np.ndarray): First embedding
        embedding2 (np.ndarray): Second embedding
        
    Returns:
        float: Cosine similarity distance (1 - similarity)
    """
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return 1 - similarity  # Convert similarity to distance


def findNearestNeihbors(keyword: str, k: int = 3) -> List[Tuple[str, float, np.ndarray]]:
    """
    Find k nearest neighbors using embedding similarity.
    
    Args:
        keyword (str): Keyword to find neighbors for
        k (int): Number of neighbors to find
        
    Returns:
        List[Tuple[str, float, np.ndarray]]: List of (keyword, distance, chatgpt_trend) tuples
    """
    global training_data
    
    # Get embedding for the query keyword
    queryEmbedding = getEmbedding(keyword)
    
    distances = []
    
    for trainKeyword, (trainEmbedding, trainChatgpt) in training_data.items():
        # Calculate cosine similarity distance
        distance = cosineSimilarityDistance(queryEmbedding, trainEmbedding)
        distances.append((trainKeyword, distance, trainChatgpt))
    
    # Sort by distance and return top k
    distances.sort(key=lambda x: x[1])
    return distances[:k]


def weightedAveragePrediction(neighbors: List[Tuple[str, float, np.ndarray]], weights: str = 'distance') -> np.ndarray:
    """
    Calculate weighted average prediction from neighbors.
    
    Args:
        neighbors (List[Tuple[str, float, np.ndarray]]): List of (keyword, distance, chatgpt_trend)
        weights (str): Weighting scheme ('uniform' or 'distance')
        
    Returns:
        np.ndarray: Predicted ChatGPT trend
    """
    if weights == 'uniform':
        # Simple average
        predictions = np.array([neighbor[2] for neighbor in neighbors])
        return np.mean(predictions, axis=0)
    
    elif weights == 'distance':
        # Weighted average based on inverse distance
        weightsList = []
        predictions = []
        
        for keyword, distance, chatgptTrend in neighbors:
            # Avoid division by zero
            weight = 1.0 / (distance + 1e-8)
            weightsList.append(weight)
            predictions.append(chatgptTrend)
        
        weightsArray = np.array(weightsList)
        predictions = np.array(predictions)
        
        # Normalize weights
        weightsArray = weightsArray / np.sum(weightsArray)
        
        # Calculate weighted average
        return np.average(predictions, axis=0, weights=weightsArray)


def generateSyntheticsTrendGPT(googleTrend: np.ndarray, keyword: str) -> np.ndarray:
    """
    Generate synthetic ChatGPT trend based on Google trend.
    In a real scenario, this would be replaced with actual ChatGPT trend data.
    
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


def fitModel(trainingKeywords: List[str], startDate: Optional[str] = None, endDate: Optional[str] = None, model: str = 'text-embedding-ada-002'):
    """
    Fit the model by collecting training data and embeddings.
    
    Args:
        trainingKeywords (List[str]): List of keywords to use for training
        startDate (str, optional): Start date for data collection
        endDate (str, optional): End date for data collection
        model (str): OpenAI embedding model to use
    """
    global training_data, is_fitted
    
    logger.info(f"Starting model fitting with {len(trainingKeywords)} keywords")
    logger.info(f"Date range: {startDate} to {endDate}")
    logger.info(f"Using embedding model: {model}")
    
    print(f"Collecting training data and embeddings for {len(trainingKeywords)} keywords...")
    
    successfulKeywords = 0
    for i, keyword in enumerate(trainingKeywords):
        try:
            logger.info(f"Processing keyword {i+1}/{len(trainingKeywords)}: '{keyword}'")
            print(f"Processing {i+1}/{len(trainingKeywords)}: {keyword}")
            
            # Get embedding for the keyword
            embedding = getEmbedding(keyword, model)
            
            # Get Google Trends data with fallback to random data
            logger.info(f"Fetching Google Trends data for '{keyword}'")
            googleTrend = None
            googleDf = None
            
            try:
                googleDf = getGoogleTrends(keyword, startDate, endDate)
                googleTrend = googleDf['trendValue'].values
                logger.info(f"✅ Fetched Google Trends data for '{keyword}'")
            except Exception as trend_error:
                if "429" in str(trend_error) or "Too Many Requests" in str(trend_error):
                    logger.warning(f"Rate limited for '{keyword}', generating random Google Trends data")
                else:
                    logger.warning(f"Error fetching Google Trends for '{keyword}': {trend_error}")
                    logger.info(f"Generating random Google Trends data instead")
                googleTrend = None
            
            # If Google Trends API failed, generate random data
            if googleTrend is None:
                # Calculate number of days between start and end date
                from datetime import datetime
                start_dt = datetime.strptime(startDate, "%Y-%m-%d") if startDate else datetime.now()
                end_dt = datetime.strptime(endDate, "%Y-%m-%d") if endDate else datetime.now()
                days = (end_dt - start_dt).days + 1
                
                # Generate random Google Trends data
                np.random.seed(hash(keyword) % 2**32)
                base_trend = np.random.uniform(20, 80, days)
                trend_bias = np.random.uniform(-0.5, 0.5)
                trend_line = np.linspace(0, trend_bias * days, days)
                weekly_pattern = np.sin(np.arange(days) * 2 * np.pi / 7) * 10
                noise = np.random.normal(0, 8, days)
                googleTrend = np.clip(base_trend + trend_line + weekly_pattern + noise, 0, 100).astype(int)
                logger.info(f"🔄 Generated random Google Trends data for '{keyword}' ({days} days)")
                
                # Create a proper DataFrame structure
                googleDf = pd.DataFrame({
                    'date': pd.date_range(start=startDate or '2025-06-01', periods=days, freq='D'),
                    'trendValue': googleTrend,
                    'isPartial': [False] * days,
                    'keyword': [keyword] * days
                })
            
            # Generate synthetic ChatGPT trend (in real scenario, this would be actual ChatGPT data)
            logger.info(f"Generating synthetic ChatGPT trend for '{keyword}'")
            chatgptTrend = generateSyntheticsTrendGPT(googleTrend, keyword)
            
            # Store the embedding and ChatGPT trend pair
            training_data[keyword] = (embedding, chatgptTrend)
            successfulKeywords += 1
            
            logger.info(f"Successfully processed keyword '{keyword}'")
            
        except Exception as e:
            logger.error(f"Error processing keyword '{keyword}': {e}")
            print(f"Error processing keyword '{keyword}': {e}")
            continue
    
    is_fitted = True
    logger.info(f"Model fitting completed. Successfully processed {successfulKeywords}/{len(trainingKeywords)} keywords")
    print(f"Training completed. Collected data for {len(training_data)} keywords.")
    
    # Save training data
    saveTrainingData()


def predictTrendGPT(keyword: str, k: int = 3, weights: str = 'distance',  startDate: Optional[str] = None, endDate: Optional[str] = None) -> Tuple[np.ndarray, List[Tuple[str, float]]]:
    """
    Predict ChatGPT trend for a new keyword.
    
    Args:
        keyword (str): Keyword to predict for
        k (int): Number of nearest neighbors to consider
        weights (str): Weighting scheme ('uniform' or 'distance')
        startDate (str, optional): Start date for data collection
        endDate (str, optional): End date for data collection
        
    Returns:
        Tuple[np.ndarray, List[Tuple[str, float]]]: (predicted_trend, neighbors_info)
    """
    global is_fitted
    
    logger.info(f"Making prediction for keyword: '{keyword}' with k={k}, weights='{weights}'")
    
    if not is_fitted:
        logger.error("Model not fitted. Cannot make predictions.")
        raise ValueError("Model must be fitted before making predictions. Call fit_model() first.")
    
    # Find nearest neighbors using embeddings
    logger.info(f"Finding {k} nearest neighbors for '{keyword}'")
    neighbors = findNearestNeihbors(keyword, k)
    
    # Extract neighbor info for return
    neighborsInfo = [(neighbor[0], neighbor[1]) for neighbor in neighbors]
    
    # Log neighbor information
    neighborKeywords = [n[0] for n in neighborsInfo]
    neighborDistances = [n[1] for n in neighborsInfo]
    logger.info(f"Found neighbors for '{keyword}': {neighborKeywords}")
    logger.info(f"Neighbor distances: {neighborDistances}")
    
    # Make prediction
    logger.info(f"Calculating weighted average prediction with weights='{weights}'")
    predictedTrend = weightedAveragePrediction(neighbors, weights)
    
    logger.info(f"Prediction completed for '{keyword}'. Average trend value: {np.mean(predictedTrend):.2f}")
    
    return predictedTrend, neighborsInfo


def getSemanticSimilarityMatrix(keywords: List[str], model: str = 'text-embedding-ada-002') -> pd.DataFrame:
    """
    Get semantic similarity matrix between keywords using embeddings.
    
    Args:
        keywords (List[str]): List of keywords to compare
        model (str): OpenAI embedding model to use
        
    Returns:
        pd.DataFrame: Similarity matrix
    """
    embeddings = []
    for keyword in keywords:
        embedding = getEmbedding(keyword, model)
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    similarityMatrix = cosine_similarity(embeddings)
    
    return pd.DataFrame(similarityMatrix, index=keywords, columns=keywords)


def evaluateModel(testKeywords: List[str], k: int = 3, weights: str = 'distance', startDate: Optional[str] = None, endDate: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate the model on test keywords.
    
    Args:
        testKeywords (List[str]): List of keywords to test
        k (int): Number of nearest neighbors to consider
        weights (str): Weighting scheme ('uniform' or 'distance')
        startDate (str, optional): Start date for data collection
        endDate (str, optional): End date for data collection
        
    Returns:
        Dict[str, float]: Evaluation metrics
    """
    global is_fitted
    
    logger.info(f"Starting model evaluation with {len(testKeywords)} test keywords")
    logger.info(f"Evaluation parameters: k={k}, weights='{weights}'")
    
    if not is_fitted:
        logger.error("Model not fitted. Cannot evaluate.")
        raise ValueError("Model must be fitted before evaluation. Call fit_model() first.")
    
    predictions = []
    actuals = []
    successfulEvaluations = 0
    
    for i, keyword in enumerate(testKeywords):
        try:
            logger.info(f"Evaluating keyword {i+1}/{len(testKeywords)}: '{keyword}'")
            
            # Get actual data with fallback to random data
            googleTrend = None
            googleDf = None
            
            try:
                googleDf = getGoogleTrends(keyword, startDate, endDate)
                googleTrend = googleDf['trendValue'].values
                logger.info(f"✅ Fetched Google Trends data for '{keyword}' evaluation")
            except Exception as trend_error:
                if "429" in str(trend_error) or "Too Many Requests" in str(trend_error):
                    logger.warning(f"Rate limited for '{keyword}' evaluation, generating random Google Trends data")
                else:
                    logger.warning(f"Error fetching Google Trends for '{keyword}' evaluation: {trend_error}")
                    logger.info(f"Generating random Google Trends data instead")
                googleTrend = None
            
            # If Google Trends API failed, generate random data
            if googleTrend is None:
                # Calculate number of days between start and end date
                from datetime import datetime
                start_dt = datetime.strptime(startDate, "%Y-%m-%d") if startDate else datetime.now()
                end_dt = datetime.strptime(endDate, "%Y-%m-%d") if endDate else datetime.now()
                days = (end_dt - start_dt).days + 1
                
                # Generate random Google Trends data
                np.random.seed(hash(keyword) % 2**32)
                base_trend = np.random.uniform(20, 80, days)
                trend_bias = np.random.uniform(-0.5, 0.5)
                trend_line = np.linspace(0, trend_bias * days, days)
                weekly_pattern = np.sin(np.arange(days) * 2 * np.pi / 7) * 10
                noise = np.random.normal(0, 8, days)
                googleTrend = np.clip(base_trend + trend_line + weekly_pattern + noise, 0, 100).astype(int)
                logger.info(f"🔄 Generated random Google Trends data for '{keyword}' evaluation ({days} days)")
                
                # Create a proper DataFrame structure
                googleDf = pd.DataFrame({
                    'date': pd.date_range(start=startDate or '2025-06-01', periods=days, freq='D'),
                    'trendValue': googleTrend,
                    'isPartial': [False] * days,
                    'keyword': [keyword] * days
                })
            
            actualChatgpt = generateSyntheticsTrendGPT(googleTrend, keyword)
            
            # Make prediction
            predictedChatgpt, _ = predictTrendGPT(keyword, k, weights, startDate, endDate)
            
            predictions.append(predictedChatgpt)
            actuals.append(actualChatgpt)
            successfulEvaluations += 1
            
            logger.info(f"Successfully evaluated keyword '{keyword}'")
            
        except Exception as e:
            logger.error(f"Error evaluating keyword '{keyword}': {e}")
            print(f"Error evaluating keyword '{keyword}': {e}")
            continue
    
    if not predictions:
        logger.error("No successful predictions for evaluation")
        raise ValueError("No successful predictions for evaluation.")
    
    logger.info(f"Evaluation completed. Successfully evaluated {successfulEvaluations}/{len(testKeywords)} keywords")
    
    # Calculate metrics
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    mse = mean_squared_error(actuals.flatten(), predictions.flatten())
    mae = mean_absolute_error(actuals.flatten(), predictions.flatten())
    r2 = r2_score(actuals.flatten(), predictions.flatten())
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }
    
    logger.info(f"Evaluation metrics: MSE={mse:.4f}, MAE={mae:.4f}, R²={r2:.4f}, RMSE={rmse:.4f}")
    
    return metrics


def plotPrediction(keyword: str, k: int = 3, weights: str = 'distance', startDate: Optional[str] = None, endDate: Optional[str] = None, savePath: Optional[str] = None):
    """
    Plot prediction results for a keyword.
    
    Args:
        keyword (str): Keyword to plot
        k (int): Number of nearest neighbors to consider
        weights (str): Weighting scheme ('uniform' or 'distance')
        startDate (str, optional): Start date
        endDate (str, optional): End date
        savePath (str, optional): Path to save the plot
    """
    # Get data with fallback to random data
    googleTrend = None
    googleDf = None
    
    try:
        googleDf = getGoogleTrends(keyword, startDate, endDate)
        googleTrend = googleDf['trendValue'].values
        print(f"✅ Fetched Google Trends data for '{keyword}' plotting")
    except Exception as trend_error:
        if "429" in str(trend_error) or "Too Many Requests" in str(trend_error):
            print(f"⚠️ Rate limited for '{keyword}' plotting, generating random Google Trends data")
        else:
            print(f"⚠️ Error fetching Google Trends for '{keyword}' plotting: {trend_error}")
            print(f"Generating random Google Trends data instead")
        googleTrend = None
    
    # If Google Trends API failed, generate random data
    if googleTrend is None:
        # Calculate number of days between start and end date
        from datetime import datetime
        start_dt = datetime.strptime(startDate, "%Y-%m-%d") if startDate else datetime.now()
        end_dt = datetime.strptime(endDate, "%Y-%m-%d") if endDate else datetime.now()
        days = (end_dt - start_dt).days + 1
        
        # Generate random Google Trends data
        np.random.seed(hash(keyword) % 2**32)
        base_trend = np.random.uniform(20, 80, days)
        trend_bias = np.random.uniform(-0.5, 0.5)
        trend_line = np.linspace(0, trend_bias * days, days)
        weekly_pattern = np.sin(np.arange(days) * 2 * np.pi / 7) * 10
        noise = np.random.normal(0, 8, days)
        googleTrend = np.clip(base_trend + trend_line + weekly_pattern + noise, 0, 100).astype(int)
        print(f"🔄 Generated random Google Trends data for '{keyword}' plotting ({days} days)")
        
        # Create a mock DataFrame for plotting
        googleDf = pd.DataFrame({
            'date': pd.date_range(start=startDate or '2025-06-01', periods=days, freq='D'),
            'trendValue': googleTrend
        })
    
    actualChatgpt = generateSyntheticsTrendGPT(googleTrend, keyword)
    predictedChatgpt, neighborsInfo = predictTrendGPT(keyword, k, weights, startDate, endDate)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    # Plot Google Trends
    plt.subplot(3, 1, 1)
    plt.plot(googleDf['date'], googleTrend, 'b-', label='Google Trends', linewidth=2)
    plt.title(f'Google Trends for "{keyword}"')
    plt.ylabel('Trend Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot ChatGPT Trends
    plt.subplot(3, 1, 2)
    plt.plot(googleDf['date'], actualChatgpt, 'g-', label='Actual ChatGPT Trends', linewidth=2)
    plt.plot(googleDf['date'], predictedChatgpt, 'r--', label='Predicted ChatGPT Trends', linewidth=2)
    plt.title(f'ChatGPT Trends Prediction for "{keyword}"')
    plt.ylabel('Trend Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot similarity scores
    plt.subplot(3, 1, 3)
    neighborKeywords = [n[0] for n in neighborsInfo]
    neighborSimilarities = [1 - n[1] for n in neighborsInfo]  # Convert distance to similarity
    
    bars = plt.bar(neighborKeywords, neighborSimilarities, color='skyblue', alpha=0.7)
    plt.title('Semantic Similarity with Nearest Neighbors')
    plt.ylabel('Similarity Score')
    plt.xlabel('Neighbor Keywords')
    plt.ylim(0, 1)
    
    # Add similarity values on bars
    for bar, similarity in zip(bars, neighborSimilarities):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{similarity:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if savePath:
        plt.savefig(savePath, dpi=300, bbox_inches='tight')
    
    plt.show()


def plotEmbeddingSpace(keywords: List[str], model: str = 'text-embedding-ada-002', savePath: Optional[str] = None):
    """
    Visualize keywords in embedding space using PCA.
    
    Args:
        keywords (List[str]): Keywords to visualize
        model (str): OpenAI embedding model to use
        savePath (str, optional): Path to save the plot
    """
    try:        
        # Get embeddings for keywords
        embeddings = []
        for keyword in keywords:
            embedding = getEmbedding(keyword, model)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
        
        # Reduce to 2D using PCA
        pca = PCA(n_components=2)
        embeddings2d = pca.fit_transform(embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.scatter(embeddings2d[:, 0], embeddings2d[:, 1], alpha=0.7, s=100)
        
        # Add labels
        for i, keyword in enumerate(keywords):
            plt.annotate(keyword, (embeddings2d[i, 0], embeddings2d[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=10)
        
        plt.title('Keyword Embeddings in 2D Space (PCA)')
        plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.grid(True, alpha=0.3)
        
        if savePath:
            plt.savefig(savePath, dpi=300, bbox_inches='tight')
        
        plt.show()
        
    except ImportError:
        print("sklearn.decomposition.PCA not available for embedding visualization")


def resetModel():
    """
    Reset the model state (clear training data and embeddings cache).
    """
    global training_data, keyword_embeddings, is_fitted
    
    logger.info("Resetting model state")
    logger.info(f"Clearing training data ({len(training_data)} keywords)")
    logger.info(f"Clearing embeddings cache ({len(keyword_embeddings)} embeddings)")
    
    training_data = {}
    keyword_embeddings = {}
    is_fitted = False
    
    logger.info("Model state reset completed")
    print("Model state reset. Training data and embeddings cache cleared.")