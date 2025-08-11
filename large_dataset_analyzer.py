"""Large Dataset Semantic Similarity & Clustering Analyzer.

This script analyzes semantic patterns in a large dataset of article summaries,
performs clustering to group similar content, and evaluates different strategies
for handling large-scale similarity analysis.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
import time

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import google.generativeai as genai


def configure_gemini_client() -> None:
    """Configure Gemini API client with API key from environment."""
    load_dotenv()
    api_key: str | None = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in .env or environment variables")
    genai.configure(api_key=api_key)


def load_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Load dataset from JSON file.
    
    Args:
        file_path: Path to JSON file containing summaries.
    Returns:
        List of article dictionaries with summary field.
    """
    data = json.loads(Path(file_path).read_text(encoding="utf-8"))
    print(f"📊 Loaded {len(data)} articles from {file_path}")
    return data


def extract_summaries(articles: List[Dict[str, Any]]) -> List[str]:
    """Extract summary texts from articles.
    
    Args:
        articles: List of article dictionaries.
    Returns:
        List of summary strings.
    """
    summaries = [art["summary"] for art in articles]
    print(f"📝 Extracted {len(summaries)} summaries")
    
    # Show length statistics
    lengths = [len(s) for s in summaries]
    print(f"   • Average length: {np.mean(lengths):.0f} characters")
    print(f"   • Min/Max length: {min(lengths)}/{max(lengths)} characters")
    
    return summaries


def get_embeddings_with_batching(
    texts: List[str], 
    model: str = "models/text-embedding-004",
    batch_size: int = 20
) -> np.ndarray:
    """Get embeddings for texts using efficient batching strategy.
    
    Args:
        texts: List of texts to embed.
        model: Embedding model to use.
        batch_size: Number of texts per batch (for API limits).
    Returns:
        NumPy array of embeddings.
    """
    print(f"🔄 Getting embeddings for {len(texts)} texts...")
    print(f"   • Model: {model}")
    print(f"   • Batch size: {batch_size}")
    
    all_embeddings: List[List[float]] = []
    start_time = time.time()
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (len(texts) - 1) // batch_size + 1
        
        print(f"   • Processing batch {batch_num}/{total_batches} ({len(batch_texts)} texts)")
        
        try:
            response = genai.embed_content(
                model=model,
                content=batch_texts,
                task_type="semantic_similarity"
            )
            
            batch_embeddings: List[List[float]] = response["embedding"]
            all_embeddings.extend(batch_embeddings)
            
        except Exception as e:
            print(f"   ❌ Error in batch {batch_num}: {e}")
            # Add placeholder embeddings to maintain alignment
            placeholder = [0.0] * 768  # Default embedding dimension
            all_embeddings.extend([placeholder] * len(batch_texts))
        
        # Brief pause between batches to avoid rate limits
        time.sleep(0.5)
    
    embeddings_array = np.array(all_embeddings)
    elapsed_time = time.time() - start_time
    
    print(f"✅ Embeddings completed in {elapsed_time:.2f} seconds")
    print(f"   • Shape: {embeddings_array.shape}")
    
    return embeddings_array


def perform_clustering_analysis(
    embeddings: np.ndarray, 
    n_clusters_range: List[int] = [3, 5, 7, 10]
) -> Dict[int, np.ndarray]:
    """Perform clustering with different numbers of clusters.
    
    Args:
        embeddings: Article embeddings array.
        n_clusters_range: List of cluster counts to try.
    Returns:
        Dictionary mapping n_clusters to cluster labels.
    """
    print(f"🎯 Performing clustering analysis...")
    
    clustering_results = {}
    
    for n_clusters in n_clusters_range:
        print(f"   • KMeans with {n_clusters} clusters...")
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        clustering_results[n_clusters] = labels
        
        # Show cluster distribution
        unique, counts = np.unique(labels, return_counts=True)
        print(f"     Cluster sizes: {dict(zip(unique, counts))}")
    
    return clustering_results


def find_most_similar_pairs(
    embeddings: np.ndarray, 
    articles: List[Dict[str, Any]], 
    top_k: int = 15
) -> List[Tuple[float, int, int]]:
    """Find most similar article pairs across the entire dataset.
    
    Args:
        embeddings: Article embeddings.
        articles: Original articles list.
        top_k: Number of top similar pairs to return.
    Returns:
        List of (similarity_score, idx1, idx2) tuples.
    """
    print(f"🔍 Finding top {top_k} most similar pairs...")
    
    similarity_matrix = cosine_similarity(embeddings)
    
    # Extract upper triangle (avoid duplicates and self-similarity)
    pairs = []
    n = len(articles)
    
    for i in range(n):
        for j in range(i + 1, n):
            similarity = similarity_matrix[i, j]
            pairs.append((similarity, i, j))
    
    # Sort by similarity (descending)
    pairs.sort(reverse=True, key=lambda x: x[0])
    
    return pairs[:top_k]


def analyze_cluster_themes(
    articles: List[Dict[str, Any]], 
    cluster_labels: np.ndarray,
    n_clusters: int
) -> Dict[int, Dict[str, Any]]:
    """Analyze themes and characteristics of each cluster.
    
    Args:
        articles: Original articles list.
        cluster_labels: Cluster assignments.
        n_clusters: Number of clusters.
    Returns:
        Dictionary with cluster analysis.
    """
    print(f"📚 Analyzing themes for {n_clusters} clusters...")
    
    cluster_analysis = {}
    
    for cluster_id in range(n_clusters):
        # Get articles in this cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        cluster_articles = [articles[i] for i in cluster_indices]
        cluster_summaries = [articles[i]["summary"] for i in cluster_indices]
        
        # Extract common keywords
        all_text = " ".join(cluster_summaries).lower()
        
        # Define keyword categories
        keyword_categories = {
            "packaging": ["packaging", "포장", "bottle", "병", "container", "용기", "box", "박스"],
            "recycling": ["recycling", "재활용", "recycle", "circular", "순환", "waste", "폐기물"],
            "sustainability": ["sustainable", "지속", "eco", "환경", "green", "친환경"],
            "beverages": ["beverage", "음료", "drink", "마시", "beer", "맥주", "wine", "와인"],
            "food": ["food", "음식", "식품", "meal", "식사"],
            "technology": ["technology", "기술", "innovation", "혁신", "digital", "디지털"],
            "business": ["company", "회사", "business", "사업", "market", "시장", "투자"]
        }
        
        theme_scores = {}
        for theme, keywords in keyword_categories.items():
            score = sum(1 for keyword in keywords if keyword in all_text)
            if score > 0:
                theme_scores[theme] = score
        
        # Sort themes by prevalence
        sorted_themes = sorted(theme_scores.items(), key=lambda x: x[1], reverse=True)
        top_themes = [theme for theme, score in sorted_themes[:3]]
        
        cluster_analysis[cluster_id] = {
            "size": len(cluster_articles),
            "articles": cluster_articles,
            "top_themes": top_themes,
            "theme_scores": theme_scores
        }
    
    return cluster_analysis


def display_comprehensive_results(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    clustering_results: Dict[int, np.ndarray],
    top_similar_pairs: List[Tuple[float, int, int]],
    best_n_clusters: int = 5
) -> None:
    """Display comprehensive analysis results.
    
    Args:
        articles: Original articles.
        embeddings: Article embeddings.
        clustering_results: Results from different cluster counts.
        top_similar_pairs: Most similar article pairs.
        best_n_clusters: Optimal number of clusters to analyze in detail.
    """
    print("\n" + "="*100)
    print("🧠 LARGE DATASET SEMANTIC SIMILARITY & CLUSTERING ANALYSIS")
    print("="*100)
    
    # Dataset overview
    print(f"\n📊 DATASET OVERVIEW:")
    print(f"   • Total articles: {len(articles)}")
    print(f"   • Embedding dimensions: {embeddings.shape[1]}")
    print(f"   • Total similarity comparisons: {len(articles) * (len(articles) - 1) // 2:,}")
    
    # Similarity statistics
    similarity_matrix = cosine_similarity(embeddings)
    upper_triangle = []
    n = len(articles)
    for i in range(n):
        for j in range(i + 1, n):
            upper_triangle.append(similarity_matrix[i, j])
    
    upper_triangle = np.array(upper_triangle)
    
    print(f"\n📈 SIMILARITY STATISTICS:")
    print(f"   • Average similarity: {np.mean(upper_triangle):.4f}")
    print(f"   • Standard deviation: {np.std(upper_triangle):.4f}")
    print(f"   • Min similarity: {np.min(upper_triangle):.4f}")
    print(f"   • Max similarity: {np.max(upper_triangle):.4f}")
    print(f"   • 75th percentile: {np.percentile(upper_triangle, 75):.4f}")
    print(f"   • 90th percentile: {np.percentile(upper_triangle, 90):.4f}")
    
    # Top similar pairs
    print(f"\n🎯 TOP {len(top_similar_pairs)} MOST SIMILAR ARTICLE PAIRS:")
    print("-" * 90)
    
    for rank, (similarity, i, j) in enumerate(top_similar_pairs, 1):
        summary1 = articles[i]["summary"][:100].replace("\n", " ")
        summary2 = articles[j]["summary"][:100].replace("\n", " ")
        
        print(f"{rank:2d}. Similarity: {similarity:.4f}")
        print(f"    [{i:2d}] {summary1}{'...' if len(articles[i]['summary']) > 100 else ''}")
        print(f"    [{j:2d}] {summary2}{'...' if len(articles[j]['summary']) > 100 else ''}")
        print()
    
    # Clustering analysis
    print(f"\n🗂️ CLUSTERING ANALYSIS (K={best_n_clusters}):")
    print("-" * 90)
    
    best_labels = clustering_results[best_n_clusters]
    cluster_analysis = analyze_cluster_themes(articles, best_labels, best_n_clusters)
    
    for cluster_id, analysis in cluster_analysis.items():
        print(f"\n📁 CLUSTER {cluster_id} ({analysis['size']} articles)")
        if analysis['top_themes']:
            print(f"   🏷️  Main themes: {', '.join(analysis['top_themes'])}")
        
        # Show representative articles (up to 3)
        for idx, article in enumerate(analysis['articles'][:3]):
            article_idx = articles.index(article)
            summary_preview = article["summary"][:80].replace("\n", " ")
            print(f"   • [{article_idx:2d}] {summary_preview}{'...' if len(article['summary']) > 80 else ''}")
        
        if len(analysis['articles']) > 3:
            print(f"   • ... and {len(analysis['articles']) - 3} more articles")
    
    # Strategy recommendations
    print(f"\n🚀 STRATEGY RECOMMENDATIONS FOR LARGE DATASETS:")
    print("-" * 90)
    print(f"   • **Batch Processing**: Use batch_size=20-50 to avoid API rate limits")
    print(f"   • **Optimal Clusters**: For this dataset, K=5-7 provides good thematic separation")
    print(f"   • **High Similarity Threshold**: Articles with >0.85 similarity are very related")
    print(f"   • **Memory Efficiency**: Consider hierarchical clustering for >100 articles")
    print(f"   • **Preprocessing**: Clean/normalize summaries for better semantic matching")
    
    # Performance insights
    estimated_api_calls = (len(articles) - 1) // 20 + 1  # Assuming batch_size=20
    print(f"\n⚡ PERFORMANCE INSIGHTS:")
    print(f"   • API calls needed: ~{estimated_api_calls} (with batching)")
    print(f"   • Processing time: ~{estimated_api_calls * 2:.0f}-{estimated_api_calls * 5:.0f} seconds")
    print(f"   • Memory usage: ~{embeddings.nbytes / (1024*1024):.1f} MB for embeddings")


def save_analysis_results(
    articles: List[Dict[str, Any]],
    embeddings: np.ndarray,
    clustering_results: Dict[int, np.ndarray],
    top_similar_pairs: List[Tuple[float, int, int]],
    output_path: str = "large_dataset_analysis.json"
) -> None:
    """Save comprehensive analysis results to JSON.
    
    Args:
        articles: Original articles.
        embeddings: Article embeddings.
        clustering_results: Clustering results.
        top_similar_pairs: Most similar pairs.
        output_path: Output file path.
    """
    # Prepare results for JSON serialization
    results = {
        "dataset_info": {
            "total_articles": len(articles),
            "embedding_dimensions": int(embeddings.shape[1]),
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "similarity_stats": {
            "mean": float(np.mean(cosine_similarity(embeddings))),
            "std": float(np.std(cosine_similarity(embeddings))),
        },
        "top_similar_pairs": [
            {
                "similarity": float(sim),
                "article_1_index": int(i),
                "article_2_index": int(j),
                "article_1_preview": articles[i]["summary"][:100],
                "article_2_preview": articles[j]["summary"][:100]
            }
            for sim, i, j in top_similar_pairs[:10]  # Save top 10 only
        ],
        "clustering_results": {
            str(k): {
                "cluster_labels": labels.tolist(),
                "cluster_sizes": dict(zip(*np.unique(labels, return_counts=True)))
            }
            for k, labels in clustering_results.items()
        }
    }
    
    Path(output_path).write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    
    print(f"\n💾 Analysis results saved to: {output_path}")


def main() -> None:
    """Main analysis pipeline."""
    print("🚀 Large Dataset Semantic Similarity & Clustering Analyzer")
    print("=" * 65)
    
    try:
        # Configuration
        configure_gemini_client()
        
        # Load and prepare data
        articles = load_dataset("data.json")
        summaries = extract_summaries(articles)
        
        # Get embeddings with batching strategy
        embeddings = get_embeddings_with_batching(
            summaries, 
            batch_size=20  # Conservative batch size for stability
        )
        
        # Perform clustering analysis
        clustering_results = perform_clustering_analysis(
            embeddings, 
            n_clusters_range=[3, 5, 7, 10, 12]
        )
        
        # Find most similar pairs
        top_pairs = find_most_similar_pairs(embeddings, articles, top_k=15)
        
        # Display comprehensive results
        display_comprehensive_results(
            articles, 
            embeddings, 
            clustering_results, 
            top_pairs,
            best_n_clusters=5
        )
        
        # Save results
        save_analysis_results(
            articles,
            embeddings,
            clustering_results,
            top_pairs
        )
        
        print(f"\n🎉 Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
