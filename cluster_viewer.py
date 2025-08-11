"""Cluster Viewer - Show detailed groupings of similar articles.

This script loads cached embeddings and displays detailed clustering results
with actual article summaries grouped by similarity.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


def load_cached_embeddings(cache_pattern: str = "embeddings_cache_41_*.pkl") -> np.ndarray:
    """Load cached embeddings from most recent cache file."""
    cache_files = list(Path(".").glob(cache_pattern))
    if not cache_files:
        raise FileNotFoundError(f"No cache files found matching {cache_pattern}")
    
    # Get most recent cache file
    latest_cache = max(cache_files, key=lambda p: p.stat().st_mtime)
    
    with open(latest_cache, 'rb') as f:
        embeddings = pickle.load(f)
    
    print(f"📦 Loaded embeddings from: {latest_cache}")
    print(f"   • Shape: {embeddings.shape}")
    
    return embeddings


def load_articles_data(file_path: str = "data.json") -> List[Dict[str, Any]]:
    """Load articles from JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    print(f"📊 Loaded {len(articles)} articles from {file_path}")
    return articles


def perform_clustering(embeddings: np.ndarray, n_clusters: int = 3) -> Tuple[np.ndarray, KMeans]:
    """Perform KMeans clustering on embeddings."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    
    return cluster_labels, kmeans


def find_cluster_centroid_articles(
    embeddings: np.ndarray, 
    cluster_labels: np.ndarray,
    kmeans: KMeans,
    articles: List[Dict[str, Any]]
) -> Dict[int, int]:
    """Find the article closest to each cluster centroid."""
    cluster_representatives = {}
    
    for cluster_id in range(len(kmeans.cluster_centers_)):
        cluster_center = kmeans.cluster_centers_[cluster_id]
        
        # Find articles in this cluster
        cluster_mask = cluster_labels == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Find closest article to centroid
        similarities = cosine_similarity([cluster_center], cluster_embeddings)[0]
        closest_idx_in_cluster = np.argmax(similarities)
        closest_article_idx = cluster_indices[closest_idx_in_cluster]
        
        cluster_representatives[cluster_id] = closest_article_idx
    
    return cluster_representatives


def calculate_intra_cluster_similarities(
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_id: int
) -> Tuple[float, float, float]:
    """Calculate similarity statistics within a cluster."""
    cluster_mask = cluster_labels == cluster_id
    cluster_embeddings = embeddings[cluster_mask]
    
    if len(cluster_embeddings) < 2:
        return 0.0, 0.0, 0.0
    
    # Calculate pairwise similarities within cluster
    similarities = cosine_similarity(cluster_embeddings)
    
    # Get upper triangle (exclude diagonal and duplicates)
    upper_triangle = similarities[np.triu_indices_from(similarities, k=1)]
    
    return (
        float(np.mean(upper_triangle)),
        float(np.min(upper_triangle)),
        float(np.max(upper_triangle))
    )


def display_detailed_clusters(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    kmeans: KMeans,
    n_clusters: int
) -> None:
    """Display detailed clustering results with articles."""
    print(f"\n{'='*100}")
    print(f"🗂️ DETAILED CLUSTERING RESULTS (K={n_clusters})")
    print(f"{'='*100}")
    
    # Find cluster representatives
    representatives = find_cluster_centroid_articles(embeddings, cluster_labels, kmeans, articles)
    
    for cluster_id in range(n_clusters):
        cluster_mask = cluster_labels == cluster_id
        cluster_articles = [articles[i] for i in np.where(cluster_mask)[0]]
        cluster_indices = np.where(cluster_mask)[0]
        
        # Calculate intra-cluster similarity stats
        avg_sim, min_sim, max_sim = calculate_intra_cluster_similarities(
            embeddings, cluster_labels, cluster_id
        )
        
        # Representative article
        rep_idx = representatives[cluster_id]
        
        print(f"\n📁 CLUSTER {cluster_id} - {len(cluster_articles)} articles")
        print(f"   🎯 Representative: Article #{rep_idx}")
        print(f"   📊 Intra-cluster similarity: avg={avg_sim:.3f}, range=[{min_sim:.3f}, {max_sim:.3f}]")
        print(f"   {'-'*90}")
        
        # Show all articles in cluster
        for idx, article_idx in enumerate(cluster_indices):
            article = articles[article_idx]
            summary = article["summary"]
            
            # Truncate long summaries
            if len(summary) > 120:
                display_summary = summary[:120] + "..."
            else:
                display_summary = summary
            
            # Clean up summary for display
            display_summary = display_summary.replace('\n', ' ').replace('\r', ' ')
            
            # Mark representative
            marker = "👑" if article_idx == rep_idx else "  "
            
            print(f"   {marker} [{article_idx:2d}] {display_summary}")
        
        print()


def find_most_similar_pairs_detailed(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    top_k: int = 10
) -> None:
    """Find and display most similar article pairs with details."""
    print(f"\n🔍 TOP {top_k} MOST SIMILAR ARTICLE PAIRS")
    print(f"{'='*100}")
    
    similarity_matrix = cosine_similarity(embeddings)
    
    # Find all pairs with similarities
    pairs = []
    n = len(articles)
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i, j]
            pairs.append((similarity, i, j))
    
    # Sort by similarity (descending)
    pairs.sort(reverse=True, key=lambda x: x[0])
    
    for rank, (similarity, i, j) in enumerate(pairs[:top_k], 1):
        article_i = articles[i]
        article_j = articles[j]
        
        summary_i = article_i["summary"][:100].replace('\n', ' ')
        summary_j = article_j["summary"][:100].replace('\n', ' ')
        
        print(f"{rank:2d}. SIMILARITY: {similarity:.4f}")
        print(f"    📄 [{i:2d}] {summary_i}{'...' if len(article_i['summary']) > 100 else ''}")
        print(f"    📄 [{j:2d}] {summary_j}{'...' if len(article_j['summary']) > 100 else ''}")
        print()


def compare_different_k_values(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    k_values: List[int] = [3, 5, 7]
) -> None:
    """Compare clustering results for different K values."""
    print(f"\n📊 CLUSTERING COMPARISON")
    print(f"{'='*60}")
    
    for k in k_values:
        if k > len(articles):
            continue
            
        cluster_labels, _ = perform_clustering(embeddings, k)
        unique, counts = np.unique(cluster_labels, return_counts=True)
        
        # Calculate silhouette-like metric (average intra-cluster similarity)
        total_intra_sim = 0.0
        valid_clusters = 0
        
        for cluster_id in range(k):
            avg_sim, _, _ = calculate_intra_cluster_similarities(embeddings, cluster_labels, cluster_id)
            if avg_sim > 0:
                total_intra_sim += avg_sim
                valid_clusters += 1
        
        avg_quality = total_intra_sim / valid_clusters if valid_clusters > 0 else 0.0
        
        print(f"K={k:2d}: sizes={dict(zip(unique, counts))}, avg_quality={avg_quality:.3f}")


def main() -> None:
    """Main cluster viewing pipeline."""
    try:
        print("🔍 Cluster Viewer - Detailed Article Groupings")
        print("=" * 50)
        
        # Load data
        articles = load_articles_data()
        embeddings = load_cached_embeddings()
        
        if len(articles) != len(embeddings):
            raise ValueError(f"Mismatch: {len(articles)} articles vs {len(embeddings)} embeddings")
        
        # Compare different K values first
        compare_different_k_values(articles, embeddings, [3, 5, 7, 10])
        
        # Ask user for preferred K
        print(f"\nBased on the comparison above, which K value would you like to see in detail?")
        try:
            k_choice = int(input("Enter K (3, 5, 7, or 10): "))
        except (ValueError, EOFError):
            k_choice = 3  # Default
            print(f"Using default K=3")
        
        # Perform clustering with chosen K
        cluster_labels, kmeans = perform_clustering(embeddings, k_choice)
        
        # Display detailed results
        display_detailed_clusters(articles, embeddings, cluster_labels, kmeans, k_choice)
        
        # Show most similar pairs
        find_most_similar_pairs_detailed(articles, embeddings, top_k=8)
        
        print(f"\n🎉 Cluster analysis completed!")
        print(f"💡 This shows you exactly which articles are grouped together by semantic similarity.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
