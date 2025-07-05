import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the current directory to path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our functions
from embeddingsKNN import (
    promptToKeywordsAndAnalyze, 
    loadTrainingData, 
    setupOpenAI,
    createTrendlinePlot
)

# Configure page
st.set_page_config(
    page_title="ChatGPT Trend Predictor",
    page_icon="📈",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_simple_trendline_plot(individual_results, successful_keywords, prompt):
    """
    Create a simple trendline plot showing only the average predicted ChatGPT trend.
    
    Args:
        individual_results: List of individual analysis results
        successful_keywords: List of successful keywords
        prompt: Original prompt for title
    """
    try:
        # Extract time series data
        dates = []
        chatgpt_trends = []
        
        # Get the first result to extract dates
        if individual_results:
            first_result = individual_results[0]
            dates = [pd.to_datetime(item['date']) for item in first_result['google_trends']['data']]
            
            # Extract ChatGPT trends for each keyword
            for result in individual_results:
                chatgpt_trends.append(result['predicted_chatgpt_trend']['trend_values'])
        
        if not dates or not chatgpt_trends:
            st.error("❌ No time series data available for trendline plot")
            return
        
        # Convert to numpy arrays
        chatgpt_trends = np.array(chatgpt_trends)
        
        # Calculate average trend across keywords
        avg_chatgpt_trend = np.mean(chatgpt_trends, axis=0)
        std_chatgpt_trend = np.std(chatgpt_trends, axis=0)
        
        # Create the trendline plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot only the average predicted ChatGPT trendline
        ax.plot(dates, avg_chatgpt_trend, color='red', linewidth=4, 
               label='Average Predicted ChatGPT', linestyle='-')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Predicted ChatGPT Trend Value', fontsize=12)
        ax.set_title(f'Average Predicted ChatGPT Trends Over Time\nPrompt: "{prompt}"', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        # Format x-axis dates
        ax.xaxis.set_major_locator(plt.matplotlib.dates.DayLocator(interval=3))
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m/%d'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        return fig
        
    except Exception as e:
        st.error(f"❌ Error creating trendline visualization: {e}")
        return None

def main():
    # Header removed
    
    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Date range selection
    st.sidebar.subheader("📅 Date Range")
    start_date = st.sidebar.date_input(
        "Start Date",
        value=datetime(2025, 6, 1),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    )
    
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime(2025, 6, 30),
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2030, 12, 31)
    )
    
    # Model parameters
    st.sidebar.subheader("🔧 Model Parameters")
    k_neighbors = st.sidebar.slider("Number of Neighbors (k)", 1, 10, 3)
    weights = st.sidebar.selectbox("Weighting Scheme", ["distance", "uniform"])
    
    # Main content area
    st.markdown("### 📝 Enter your prompt to analyze ChatGPT trends")
    
    # Text input for prompt
    prompt = st.text_input(
        "Prompt:",
        placeholder="e.g., is the model 3 a good car?",
        help="Enter a question or topic to analyze ChatGPT trends for"
    )
    
    # Analysis button
    if st.button("🚀 Analyze Trends", type="primary"):
        if not prompt:
            st.error("❌ Please enter a prompt to analyze")
            return
        
        # Check if training data is loaded
        try:
            with st.spinner("🔄 Loading training data..."):
                loadTrainingData()
            st.success("✅ Training data loaded successfully")
        except Exception as e:
            st.error(f"❌ Error loading training data: {e}")
            st.info("💡 Make sure you have training data available")
            return
        
        # Check OpenAI setup
        try:
            with st.spinner("🔧 Setting up OpenAI..."):
                setupOpenAI()
            st.success("✅ OpenAI configured successfully")
        except Exception as e:
            st.error(f"❌ Error setting up OpenAI: {e}")
            st.info("💡 Make sure your OpenAI API key is set in environment variables")
            return
        
        # Run analysis
        with st.spinner("🤖 Analyzing trends..."):
            try:
                result = promptToKeywordsAndAnalyze(
                    prompt=prompt,
                    startDate=start_date.strftime("%Y-%m-%d"),
                    endDate=end_date.strftime("%Y-%m-%d"),
                    k=k_neighbors,
                    weights=weights
                )
                
                st.success("✅ Analysis completed successfully!")
                
                # Display results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📊 Generated Keywords")
                    for i, keyword in enumerate(result['generated_keywords'], 1):
                        st.markdown(f"**{i}.** {keyword}")
                
                with col2:
                    st.markdown("### ✅ Successfully Analyzed")
                    for i, keyword in enumerate(result['successful_keywords'], 1):
                        st.markdown(f"**{i}.** {keyword}")
                
                # Display metrics
                avg = result['average_result']
                
                st.markdown("### 📈 Analysis Metrics")
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric(
                        label="Average Google Trend",
                        value=f"{avg['average_google_trend']:.1f}",
                        delta=f"±{avg['std_google_trend']:.1f}"
                    )
                
                with metric_col2:
                    st.metric(
                        label="Average Predicted ChatGPT",
                        value=f"{avg['average_predicted_chatgpt']:.1f}",
                        delta=f"±{avg['std_predicted_chatgpt']:.1f}"
                    )
                
                with metric_col3:
                    st.metric(
                        label="Prediction Confidence",
                        value=f"{avg['average_confidence']:.3f}",
                        delta=f"±{avg['std_confidence']:.3f}"
                    )
                
                with metric_col4:
                    st.metric(
                        label="Keywords Analyzed",
                        value=f"{avg['num_keywords']}",
                        delta=""
                    )
                
                # Create and display the trendline plot
                st.markdown("### 📈 Average Predicted ChatGPT Trendline")
                
                fig = create_simple_trendline_plot(
                    result['individual_results'],
                    result['successful_keywords'],
                    prompt
                )
                
                if fig:
                    st.pyplot(fig)
                

                
            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")
                st.info("💡 Check the console for detailed error information")
    
    # Footer removed

if __name__ == "__main__":
    main() 