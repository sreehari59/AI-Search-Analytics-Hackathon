import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import sys
from dotenv import load_dotenv
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

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
        color: #2c3e50;
        border-left: 4px solid #1f77b4;
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
    .cluster-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
        color: #2c3e50;
    }
    .cluster-card h3, .cluster-card h4 {
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .cluster-card p {
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    /* Set all text to white by default */
    html, body, [data-testid="stAppViewContainer"], .stApp, .stText, .stMarkdown, .stMetric, .stExpander, .stExpanderHeader, .stExpanderContent, .stButton, .stTextInput, .stSelectbox, .stSlider, .stDateInput, .stSidebar, .stSidebarContent, .stSidebarHeader, .stSidebarFooter, .stSidebarNav, .stSidebarNavItem, .stSidebarNavItemLabel, .stSidebarNavItemIcon, .stSidebarNavItemChevron, .stSidebarNavItemActive, .stSidebarNavItemActiveLabel, .stSidebarNavItemActiveIcon, .stSidebarNavItemActiveChevron {
        color: #fff !important;
    }
    /* Override for cards and metric cards to keep text dark */
    .cluster-card, .metric-card, .success-message, .info-message {
        color: #2c3e50 !important;
    }
    .cluster-card h3, .cluster-card h4 {
        color: #1f77b4 !important;
    }
    .cluster-card p {
        color: #2c3e50 !important;
    }
</style>
""", unsafe_allow_html=True)

def load_and_analyze_clusters():
    """
    Load the clustered data and create analysis.
    """
    try:
        # Read the CSV file
        df = pd.read_csv('data/intentionClustered.csv')
        
        # Transform the clusters
        cluster_mapping = {
            0: 'consideration',
            1: 'evaluation', 
            2: 'decision'
        }
        
        # Create a new column with the transformed cluster names
        df['cluster_name'] = df['cluster'].map(cluster_mapping)
        
        # Calculate the count and percentage for each cluster
        cluster_counts = df['cluster_name'].value_counts()
        cluster_percentages = (cluster_counts / len(df)) * 100
        
        return df, cluster_counts, cluster_percentages, cluster_mapping
        
    except Exception as e:
        st.error(f"Error loading cluster data: {e}")
        return None, None, None, None

def create_cluster_pie_chart(cluster_percentages):
    """
    Create a pie chart for cluster distribution using Plotly for native Streamlit integration.
    """
    try:
        # Prepare data for Plotly
        clusters = list(cluster_percentages.index)
        percentages = list(cluster_percentages.values)
        
        # Create a DataFrame for Plotly
        df_pie = pd.DataFrame({
            'Cluster': clusters,
            'Percentage': percentages,
            'Count': [cluster_percentages[cluster] * len(cluster_percentages) / 100 for cluster in clusters]
        })
        
        # Create the pie chart using Plotly
        fig = px.pie(
            df_pie, 
            values='Percentage', 
            names='Cluster',
            title='Distribution of User Intentions by Cluster',
            color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],  # Red, Teal, Blue
            hole=0.3  # Makes it a donut chart for better look
        )
        
        # Customize the layout
        fig.update_layout(
            title={
                'text': 'Distribution of User Intentions by Cluster',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 18, 'color': '#2c3e50'}
            },
            showlegend=True,
            legend={
                'title': 'Cluster Types',
                'title_font': {'size': 14, 'color': '#2c3e50'},
                'font': {'size': 12, 'color': '#2c3e50'},
                'bgcolor': 'rgba(255,255,255,0.8)',
                'bordercolor': '#2c3e50',
                'borderwidth': 1
            },
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=500
        )
        
        # Customize the pie chart appearance
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont={'size': 14, 'color': 'white'},
            hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<br>Count: %{customdata}<extra></extra>',
            customdata=df_pie['Count'].round(0).astype(int)
        )
        
        return fig
        
    except Exception as e:
        st.error(f"Error creating pie chart: {e}")
        return None

def get_example_queries(df, cluster_name, num_examples=5):
    """
    Get example queries for a specific cluster.
    """
    try:
        cluster_data = df[df['cluster_name'] == cluster_name]
        examples = cluster_data['content'].head(num_examples).tolist()
        return examples
    except Exception as e:
        st.error(f"Error getting examples for {cluster_name}: {e}")
        return []

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
    # Remove the main header/title
    # st.markdown('<h1 class="main-header">🤖 ChatGPT Trend Predictor & Cluster Analysis</h1>', unsafe_allow_html=True)
    
    # Load cluster data
    with st.spinner("🔄 Loading cluster data..."):
        df, cluster_counts, cluster_percentages, cluster_mapping = load_and_analyze_clusters()
    
    if df is not None:
        # Display cluster analysis section
        st.markdown("## 📊 User Intention Cluster Analysis")
        
        # Two columns: left for pie chart, right for example queries
        col1, col2 = st.columns([2, 2])
        
        with col1:
            fig = create_cluster_pie_chart(cluster_percentages)
            if fig:
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            with st.expander("🔍 Example Queries by Cluster", expanded=False):
                for cluster_name in cluster_mapping.values():
                    st.markdown(f"""
                    <div class="cluster-card">
                        <h4>🎯 {cluster_name.title()} Cluster</h4>
                        <p><strong>Description:</strong> Users in this phase are {cluster_name} their options.</p>
                    </div>
                    """, unsafe_allow_html=True)
                    examples = get_example_queries(df, cluster_name, num_examples=3)
                    if examples:
                        for i, example in enumerate(examples, 1):
                            if len(example) > 150:
                                example = example[:150] + "..."
                            st.markdown(f"**{i}.** {example}")
                    else:
                        st.markdown("*No examples available*")
                    st.markdown("---")
    
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
    st.markdown("## 📝 Trend Analysis")
    st.markdown("Enter your prompt to analyze ChatGPT trends")
    
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
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 ChatGPT Trend Predictor & Cluster Analysis Tool</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main() 