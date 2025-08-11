"""Robust Large Dataset Analyzer with Advanced Rate Limit Handling.

This version includes exponential backoff, smaller batch sizes, caching,
and progressive analysis to handle API rate limits gracefully.
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import time
import random

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted


def configure_gemini_client() -> None:
    """Configure Gemini API client with API key from environment."""
    load_dotenv()
    api_key: str | None = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in .env or environment variables")
    genai.configure(api_key=api_key)


def save_embeddings_cache(embeddings: np.ndarray, file_path: str = "embeddings_cache.pkl") -> None:
    """Save embeddings to cache file."""
    with open(file_path, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"💾 Embeddings cached to {file_path}")


def load_embeddings_cache(file_path: str = "embeddings_cache.pkl") -> Optional[np.ndarray]:
    """Load embeddings from cache file if exists."""
    if Path(file_path).exists():
        with open(file_path, 'rb') as f:
            embeddings = pickle.load(f)
        print(f"📦 Loaded cached embeddings from {file_path} - Shape: {embeddings.shape}")
        return embeddings
    return None


def exponential_backoff_delay(attempt: int, base_delay: float = 2.0, max_delay: float = 300.0) -> float:
    """Calculate exponential backoff delay with jitter."""
    delay = min(base_delay * (2 ** attempt), max_delay)
    # Add random jitter to avoid thundering herd
    jitter = random.uniform(0.1, 0.3) * delay
    return delay + jitter


def get_embeddings_with_robust_rate_limiting(
    texts: List[str], 
    model: str = "models/text-embedding-004",
    initial_batch_size: int = 10,
    max_retries: int = 5,
    use_cache: bool = True
) -> np.ndarray:
    """Get embeddings with robust rate limit handling and caching.
    
    Args:
        texts: List of texts to embed.
        model: Embedding model to use.
        initial_batch_size: Starting batch size (will adapt based on rate limits).
        max_retries: Maximum retry attempts per batch.
        use_cache: Whether to use cached embeddings if available.
    Returns:
        NumPy array of embeddings.
    """
    cache_file = f"embeddings_cache_{len(texts)}_{hash(str(texts[:3]))}.pkl"
    
    # Try to load from cache first
    if use_cache:
        cached_embeddings = load_embeddings_cache(cache_file)
        if cached_embeddings is not None:
            return cached_embeddings
    
    print(f"🔄 Getting embeddings for {len(texts)} texts with rate limit protection...")
    print(f"   • Model: {model}")
    print(f"   • Initial batch size: {initial_batch_size}")
    print(f"   • Max retries per batch: {max_retries}")
    
    all_embeddings: List[List[float]] = []
    current_batch_size = initial_batch_size
    start_time = time.time()
    total_api_calls = 0
    successful_batches = 0
    failed_batches = 0
    
    i = 0
    while i < len(texts):
        batch_texts = texts[i:i + current_batch_size]
        batch_num = successful_batches + failed_batches + 1
        
        print(f"   • Batch {batch_num} ({len(batch_texts)} texts, batch_size={current_batch_size})")
        
        success = False
        for retry_attempt in range(max_retries):
            try:
                total_api_calls += 1
                
                response = genai.embed_content(
                    model=model,
                    content=batch_texts,
                    task_type="semantic_similarity"
                )
                
                batch_embeddings: List[List[float]] = response["embedding"]
                all_embeddings.extend(batch_embeddings)
                successful_batches += 1
                success = True
                
                print(f"     ✅ Success (attempt {retry_attempt + 1})")
                break
                
            except ResourceExhausted as e:
                print(f"     ⏰ Rate limit hit (attempt {retry_attempt + 1}): {e}")
                
                if retry_attempt < max_retries - 1:
                    # Calculate backoff delay
                    delay = exponential_backoff_delay(retry_attempt)
                    print(f"     💤 Waiting {delay:.1f}s before retry...")
                    time.sleep(delay)
                    
                    # Reduce batch size if multiple failures
                    if retry_attempt >= 1 and current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        print(f"     📉 Reducing batch size to {current_batch_size}")
                        # Re-slice the batch with new size
                        batch_texts = texts[i:i + current_batch_size]
                
            except Exception as e:
                print(f"     ❌ Unexpected error (attempt {retry_attempt + 1}): {e}")
                if retry_attempt < max_retries - 1:
                    delay = exponential_backoff_delay(retry_attempt, base_delay=1.0)
                    time.sleep(delay)
        
        if not success:
            print(f"     💥 Batch failed after {max_retries} attempts - using placeholder embeddings")
            # Add placeholder embeddings
            placeholder = [0.0] * 768
            all_embeddings.extend([placeholder] * len(batch_texts))
            failed_batches += 1
        
        i += len(batch_texts)
        
        # Adaptive batch size - increase if successful, keep small if failing
        if success and failed_batches == 0:
            current_batch_size = min(initial_batch_size, current_batch_size + 1)
        
        # Brief pause between batches
        time.sleep(1.0)
    
    embeddings_array = np.array(all_embeddings)
    elapsed_time = time.time() - start_time
    
    print(f"✅ Embeddings completed!")
    print(f"   • Total time: {elapsed_time:.2f} seconds")
    print(f"   • Total API calls: {total_api_calls}")
    print(f"   • Successful batches: {successful_batches}")
    print(f"   • Failed batches: {failed_batches}")
    print(f"   • Final shape: {embeddings_array.shape}")
    
    # Cache the results
    if use_cache:
        save_embeddings_cache(embeddings_array, cache_file)
    
    return embeddings_array


def quick_analysis_preview(
    articles: List[Dict[str, Any]], 
    sample_size: int = 10
) -> Tuple[List[str], List[int]]:
    """Generate a quick preview analysis with a small sample.
    
    Args:
        articles: Full articles list.
        sample_size: Number of articles to sample for preview.
    Returns:
        Tuple of (sample_summaries, sample_indices).
    """
    print(f"🔬 Quick Preview Analysis (sample size: {sample_size})")
    
    # Random sampling for diversity
    sample_indices = random.sample(range(len(articles)), min(sample_size, len(articles)))
    sample_summaries = [articles[i]["summary"] for i in sample_indices]
    
    print(f"   • Sampled articles: {sample_indices}")
    
    return sample_summaries, sample_indices


def progressive_dataset_analysis(
    articles: List[Dict[str, Any]],
    preview_size: int = 10,
    full_analysis: bool = False
) -> None:
    """Perform progressive analysis - preview first, then full if requested.
    
    Args:
        articles: Full articles list.
        preview_size: Size of preview sample.
        full_analysis: Whether to run full analysis after preview.
    """
    print("🚀 Progressive Dataset Analysis")
    print("=" * 50)
    
    # Step 1: Quick preview with small sample
    print(f"\n📋 STEP 1: Preview Analysis ({preview_size} articles)")
    sample_summaries, sample_indices = quick_analysis_preview(articles, preview_size)
    
    try:
        # Get embeddings for preview
        preview_embeddings = get_embeddings_with_robust_rate_limiting(
            sample_summaries,
            initial_batch_size=5,  # Very conservative for preview
            max_retries=3
        )
        
        # Quick clustering on preview
        if len(sample_summaries) >= 3:
            n_clusters = min(3, len(sample_summaries) // 2)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = kmeans.fit_predict(preview_embeddings)
            
            print(f"\n🎯 Preview Clustering ({n_clusters} clusters):")
            for cluster_id in range(n_clusters):
                cluster_articles = [i for i, label in enumerate(labels) if label == cluster_id]
                print(f"   • Cluster {cluster_id}: {len(cluster_articles)} articles")
                for idx in cluster_articles[:2]:  # Show up to 2 articles per cluster
                    article_idx = sample_indices[idx]
                    preview = sample_summaries[idx][:80].replace("\n", " ")
                    print(f"     [{article_idx:2d}] {preview}...")
        
        # Quick similarity check
        if len(sample_summaries) >= 2:
            sim_matrix = cosine_similarity(preview_embeddings)
            avg_similarity = np.mean(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            max_similarity = np.max(sim_matrix[np.triu_indices_from(sim_matrix, k=1)])
            
            print(f"\n📊 Preview Similarity Stats:")
            print(f"   • Average similarity: {avg_similarity:.4f}")
            print(f"   • Max similarity: {max_similarity:.4f}")
        
        print(f"\n✅ Preview analysis successful!")
        
        # Step 2: Full analysis if requested
        if full_analysis:
            print(f"\n📋 STEP 2: Full Dataset Analysis ({len(articles)} articles)")
            print("⚠️  This may take several minutes due to rate limiting...")
            
            confirm = input("Continue with full analysis? (y/n): ").lower().strip()
            if confirm == 'y':
                all_summaries = [art["summary"] for art in articles]
                
                full_embeddings = get_embeddings_with_robust_rate_limiting(
                    all_summaries,
                    initial_batch_size=5,  # Conservative batch size
                    max_retries=5
                )
                
                # Full clustering analysis
                print(f"\n🎯 Full Clustering Analysis...")
                for n_clusters in [3, 5, 7]:
                    if n_clusters <= len(articles):
                        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                        labels = kmeans.fit_predict(full_embeddings)
                        
                        unique, counts = np.unique(labels, return_counts=True)
                        print(f"   • K={n_clusters}: cluster sizes {dict(zip(unique, counts))}")
                
                # Save full results
                results = {
                    "total_articles": len(articles),
                    "embeddings_shape": list(full_embeddings.shape),
                    "analysis_completed": True,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                with open("full_analysis_results.json", "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                print(f"💾 Full analysis results saved to full_analysis_results.json")
            else:
                print("👍 Full analysis skipped")
    
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print("💡 This is likely due to API rate limits. Try again later or with a smaller sample.")


def main() -> None:
    """Main progressive analysis pipeline."""
    try:
        configure_gemini_client()
        
        # Load dataset
        data = json.loads(Path("data.json").read_text(encoding="utf-8"))
        print(f"📊 Loaded {len(data)} articles from data.json")
        
        # Progressive analysis approach
        print("\n🎯 STRATEGY: Progressive Analysis to Handle Rate Limits")
        print("   1. Preview with small sample (10 articles)")
        print("   2. Optional full analysis if preview succeeds")
        print("   3. Robust rate limiting with exponential backoff")
        print("   4. Caching to avoid re-embedding on retry")
        
        progressive_dataset_analysis(
            data, 
            preview_size=10,
            full_analysis=True
        )
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
