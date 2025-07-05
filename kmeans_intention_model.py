import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# --- Load data ---
general = pd.read_csv("chats_general.csv")
specific = pd.read_csv("chats_specific.csv")
combined = pd.concat([general, specific]).sample(frac=1, random_state=42).reset_index(drop=True)

# --- Filter prompts only ---
prompts = combined[combined["content_type"] == "prompt"].copy()

# --- TF-IDF Vectorization ---
vectorizer = TfidfVectorizer(max_df=0.9, min_df=2, stop_words="english")
X = vectorizer.fit_transform(prompts["content"])

# --- KMeans Clustering ---
kmeans = KMeans(n_clusters=3, random_state=42)
prompts["cluster"] = kmeans.fit_predict(X)

# --- Optional: Save clustered data ---
prompts.to_csv("tfidf_kmeans_clustered.csv", index=False)
print("✅ Saved clustered prompts to tfidf_kmeans_clustered.csv")

# --- Inspect top terms per cluster ---
terms = vectorizer.get_feature_names_out()
for i in range(3):
    cluster_center = kmeans.cluster_centers_[i]
    top_terms = [terms[i] for i in cluster_center.argsort()[-10:][::-1]]
    print(f"\nCluster {i} top terms:", top_terms)