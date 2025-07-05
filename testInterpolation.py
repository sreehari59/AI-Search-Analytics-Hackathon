import os
import sys
import numpy as np
import pandas as pd
from embeddingsKNN import (
    setupOpenAI, fitModel, predictTrendGPT, evaluateModel, 
    plotPrediction, getSemanticSimilarityMatrix,
    loadTrainingData, saveTrainingData, resetModel
)

def testEmbeddingKNN():
    """
    Test the embedding KNN functions for ChatGPT trends prediction.
    """
    print("Embedding KNN Functions for ChatGPT Trends")
    print("=" * 50)
    
    # Setup OpenAI (you'll need to set your OpenAI API key)
    try:
        setupOpenAI()
        print("✓ OpenAI API configured successfully")
    except ValueError as e:
        print(f"✗ Error: {e}")
        print("Please set your OpenAI API key in the OPENAI_API_KEY environment variable")
        return False
    
    # Training keywords (in real scenario, these would have both Google and ChatGPT data)
    training_keywords = [
        "artificial intelligence",
        "machine learning", 
        "deep learning",
        "neural networks",
        "data science",
        "python programming",
        "natural language processing",
        "computer vision",
        "robotics",
        "blockchain"
    ]
    
    # Test keywords
    test_keywords = [
        "chatgpt",
        "openai",
        "transformer models"
    ]
    
    # Check if training data already exists
    print("\nChecking for existing training data...")
    try:
        loadTrainingData()
        print("✓ Loaded existing training data")
    except:
        print("No existing training data found. Training new model...")
        
        # Fit the model
        print("\nTraining model...")
        try:
            fitModel(training_keywords, start_date="2023-01-01", end_date="2023-12-31")
            print("✓ Model training completed")
        except Exception as e:
            print(f"✗ Error during training: {e}")
            return False
    
    # Show semantic similarity matrix
    print("\nSemantic similarity matrix:")
    try:
        similarity_matrix = getSemanticSimilarityMatrix(training_keywords[:6])
        print(similarity_matrix.round(3))
        print("✓ Similarity matrix generated")
    except Exception as e:
        print(f"✗ Error generating similarity matrix: {e}")
    
    # Make predictions
    print("\nMaking predictions...")
    successful_predictions = 0
    for keyword in test_keywords:
        try:
            predicted_trend, neighbors = predictTrendGPT(
                keyword, k=3, weights='distance', 
                start_date="2023-01-01", end_date="2023-12-31"
            )
            print(f"\n✓ Prediction for '{keyword}':")
            print(f"  Nearest neighbors: {[n[0] for n in neighbors]}")
            print(f"  Similarity scores: {[1-n[1] for n in neighbors]}")
            print(f"  Average predicted trend value: {np.mean(predicted_trend):.2f}")
            
            # Plot prediction
            plotPrediction(keyword, k=3, weights='distance', 
                          start_date="2023-01-01", end_date="2023-12-31")
            successful_predictions += 1
            
        except Exception as e:
            print(f"✗ Error predicting for '{keyword}': {e}")
    
    print(f"\n✓ Successfully made {successful_predictions}/{len(test_keywords)} predictions")
    
    # Evaluate the model
    print("\nEvaluating model...")
    try:
        metrics = evaluateModel(test_keywords, k=3, weights='distance', 
                               start_date="2023-01-01", end_date="2023-12-31")
        print("\n✓ Evaluation Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
    except Exception as e:
        print(f"✗ Error during evaluation: {e}")
    
    return True


def testDifferentParameters():
    """
    Test the model with different parameters.
    """
    print("\n" + "="*50)
    print("Testing with Different Parameters")
    print("="*50)
    
    test_keywords = ["chatgpt", "openai"]
    
    # Test different k values
    for k in [1, 3, 5]:
        print(f"\nTesting with k={k}:")
        try:
            predicted_trend, neighbors = predictTrendGPT(
                "chatgpt", k=k, weights='distance',
                start_date="2023-01-01", end_date="2023-12-31"
            )
            print(f"  ✓ Prediction successful with k={k}")
            print(f"  Neighbors: {[n[0] for n in neighbors]}")
        except Exception as e:
            print(f"  ✗ Error with k={k}: {e}")
    
    # Test different weighting schemes
    for weights in ['uniform', 'distance']:
        print(f"\nTesting with weights='{weights}':")
        try:
            predicted_trend, neighbors = predictTrendGPT(
                "chatgpt", k=3, weights=weights,
                start_date="2023-01-01", end_date="2023-12-31"
            )
            print(f"  ✓ Prediction successful with weights='{weights}'")
        except Exception as e:
            print(f"  ✗ Error with weights='{weights}': {e}")


def testDataPersistence():
    """
    Test that data is properly saved and loaded.
    """
    print("\n" + "="*50)
    print("Testing Data Persistence")
    print("="*50)
    
    # Reset model
    print("Resetting model...")
    resetModel()
    
    # Try to load training data
    print("Loading training data...")
    try:
        loadTrainingData()
        print("✓ Training data loaded successfully")
        
        # Try to make a prediction
        predicted_trend, neighbors = predictTrendGPT(
            "chatgpt", k=3, weights='distance',
            start_date="2023-01-01", end_date="2023-12-31"
        )
        print("✓ Prediction successful after loading saved data")
        
    except Exception as e:
        print(f"✗ Error testing data persistence: {e}")


def main():
    """
    Main test function.
    """
    print("Starting Embedding KNN Tests")
    print("="*60)
    
    # Run main test
    success = testEmbeddingKNN()
    
    if success:
        # Run additional tests
        testDifferentParameters()
        testDataPersistence()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Tests failed due to setup issues.")
        print("Please check your OpenAI API key and try again.")
        print("="*60)