import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import openai
from collections import Counter
import re
from typing import List, Dict
import plotly.express as px
import plotly.graph_objects as go

def extract_keywords_with_openai(prompt: str, api_key: str) -> List[str]:
    """
    Extract keywords from a prompt using OpenAI API.
    """
    try:
        client = openai.OpenAI(api_key=api_key)
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "You are a keyword extraction expert. Extract 3-5 most important keywords (single words or short phrases) from the given text. Return only the keywords separated by commas, no explanations."
                },
                {
                    "role": "user",
                    "content": f"Extract keywords from this text: {prompt}"
                }
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        keywords_text = response.choices[0].message.content.strip()
        # Split by comma and clean up
        keywords = [kw.strip().lower() for kw in keywords_text.split(',')]
        return keywords
    
    except Exception as e:
        print(f"Error extracting keywords: {e}")
        return []

def get_most_popular_keywords(df: pd.DataFrame, cluster_name: str, top_n: int = 10) -> List[tuple]:
    """
    Get the most popular keywords for a specific cluster.
    """
    cluster_data = df[df['cluster_name'] == cluster_name]
    
    # Combine all keywords for this cluster
    all_keywords = []
    for keywords_list in cluster_data['keywords']:
        if isinstance(keywords_list, list):
            all_keywords.extend(keywords_list)
    
    # Count occurrences
    keyword_counts = Counter(all_keywords)
    
    # Return top N keywords
    return keyword_counts.most_common(top_n)

def analyze_clusters(openai_api_key: str = None):
    """
    Read the CSV file, transform clusters, create a pie chart,
    and extract the most popular keywords for each cluster.
    """
    
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
    
    # Extract keywords if OpenAI API key is provided
    if openai_api_key:
        print("Extracting keywords from prompts using OpenAI...")
        df['keywords'] = df['content'].apply(lambda x: extract_keywords_with_openai(x, openai_api_key))
        print("Keyword extraction completed!")
    else:
        print("No OpenAI API key provided. Skipping keyword extraction.")
        df['keywords'] = [[] for _ in range(len(df))]
    
    # Calculate the count and percentage for each cluster
    cluster_counts = df['cluster_name'].value_counts()
    cluster_percentages = (cluster_counts / len(df)) * 100
    
    # Print summary statistics
    print("\nCluster Analysis Summary:")
    print("=" * 40)
    print(f"Total records: {len(df)}")
    print("\nCluster Distribution:")
    for cluster, count in cluster_counts.items():
        percentage = cluster_percentages[cluster]
        print(f"{cluster}: {count} records ({percentage:.1f}%)")
    
    # Get most popular keywords for each cluster
    if openai_api_key:
        print("\nMost Popular Keywords by Cluster:")
        print("=" * 40)
        
        for cluster in cluster_mapping.values():
            print(f"\n{cluster.upper()} CLUSTER:")
            popular_keywords = get_most_popular_keywords(df, cluster, top_n=10)
            
            if popular_keywords:
                for i, (keyword, count) in enumerate(popular_keywords, 1):
                    print(f"  {i}. {keyword} ({count} occurrences)")
            else:
                print("  No keywords found")
    
    # Create the pie chart visualization using Plotly
    # Prepare data for Plotly
    clusters = list(cluster_percentages.index)
    percentages = list(cluster_percentages.values)
    
    # Create a DataFrame for Plotly
    df_pie = pd.DataFrame({
        'Cluster': clusters,
        'Percentage': percentages
    })
    
    # Create the pie chart using Plotly
    fig = px.pie(
        df_pie, 
        values='Percentage', 
        names='Cluster',
        title='Distribution of User Intentions by Cluster',
        color_discrete_sequence=['#FF6B6B', '#4ECDC4', '#45B7D1'],  # Red, Teal, Blue
        hole=0.4  # Makes it a donut chart for better look
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
        height=600
    )
    
    # Customize the pie chart appearance
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label',
        textfont={'size': 14, 'color': 'white'},
        hovertemplate='<b>%{label}</b><br>Percentage: %{percent}<extra></extra>'
    )
    
    # Add a text annotation for total records
    fig.add_annotation(
        text=f'Total Records: {len(df)}',
        x=0.5,
        y=0.5,
        font=dict(size=16, color='#2c3e50'),
        showarrow=False,
        bgcolor='rgba(255,255,255,0.8)',
        bordercolor='#2c3e50',
        borderwidth=1
    )
    
    # Save and display the chart
    try:
        fig.write_image('cluster_distribution_pie.png', scale=2, width=800, height=600)
        print("✅ Chart saved as 'cluster_distribution_pie.png'")
    except Exception as e:
        print(f"⚠️  Could not save image (Chrome required): {e}")
        print("💡 The interactive chart will still be displayed.")
    
    fig.show()
    
    return df, cluster_percentages

if __name__ == "__main__":
    # You can set your OpenAI API key here or pass it as an environment variable
    # openai_api_key = "your-api-key-here"
    openai_api_key = None  # Set to None if you don't want to use OpenAI
    
    # Run the analysis
    df, percentages = analyze_clusters(openai_api_key)
    
    print("\n" + "=" * 40)
    print("Analysis complete! Check the generated chart:")
    print("- cluster_distribution_pie.png (Pie chart)")
    
    if openai_api_key:
        print("\nTo use keyword extraction, set your OpenAI API key in the script.") 