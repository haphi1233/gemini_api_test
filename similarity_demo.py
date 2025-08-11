"""Semantic Similarity Demo using Gemini Embeddings.

This script demonstrates how to calculate cosine similarity between text embeddings
using Google Gemini text-embedding-004 model.

Usage:
    python similarity_demo.py

The script will embed three sample texts and compute pairwise similarity scores.
"""
from __future__ import annotations

import os
from typing import List

import numpy as np
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai


def configure_gemini_client() -> None:
    """Configure Gemini API client with API key from environment."""
    load_dotenv()
    api_key: str | None = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY not found in .env or environment variables")
    genai.configure(api_key=api_key)


def get_embeddings(texts: List[str], model: str = "models/text-embedding-004") -> np.ndarray:
    """Get embeddings for a list of texts.

    Args:
        texts: List of texts to embed.
        model: Embedding model to use.
    Returns:
        NumPy array of shape (n_texts, embedding_dim).
    """
    print(f"Getting embeddings for {len(texts)} texts using {model}...")
    
    # Use batch processing for multiple texts
    response = genai.embed_content(
        model=model,
        content=texts,
        task_type="semantic_similarity"
    )
    
    # Extract embeddings
    embeddings_list: List[List[float]] = response["embedding"]
    embeddings_array: np.ndarray = np.array(embeddings_list)
    
    print(f"Embeddings shape: {embeddings_array.shape}")
    return embeddings_array


def compute_similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine similarity matrix for embeddings.

    Args:
        embeddings: Array of embeddings with shape (n_texts, embedding_dim).
    Returns:
        Similarity matrix with shape (n_texts, n_texts).
    """
    return cosine_similarity(embeddings)


def display_similarity_results(texts: List[str], similarity_matrix: np.ndarray) -> None:
    """Display pairwise similarity scores in a readable format.

    Args:
        texts: Original texts.
        similarity_matrix: Computed similarity matrix.
    """
    print("\n" + "="*80)
    print("SEMANTIC SIMILARITY RESULTS")
    print("="*80)
    
    # Show all pairwise comparisons (including diagonal and symmetric)
    for i, text1 in enumerate(texts):
        print(f"\n[{i+1}] \"{text1[:60]}{'...' if len(text1) > 60 else ''}\"")
        for j, text2 in enumerate(texts):
            if i != j:  # Skip self-comparison
                similarity: float = similarity_matrix[i, j]
                print(f"  → vs [{j+1}]: {similarity:.4f}")
    
    print("\n" + "="*80)
    print("TOP SIMILARITIES (excluding self-comparisons)")
    print("="*80)
    
    # Find and display unique pairs sorted by similarity
    pairs = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):  # Only upper triangle to avoid duplicates
            similarity = similarity_matrix[i, j]
            pairs.append((similarity, i, j))
    
    # Sort by similarity (descending)
    pairs.sort(reverse=True)
    
    for similarity, i, j in pairs:
        text1_short = texts[i][:40] + "..." if len(texts[i]) > 40 else texts[i]
        text2_short = texts[j][:40] + "..." if len(texts[j]) > 40 else texts[j]
        print(f"{similarity:.4f} | [{i+1}] \"{text1_short}\"")
        print(f"       | [{j+1}] \"{text2_short}\"")
        print()


def main() -> None:
    """Main demo function."""
    print("Gemini Embeddings - Semantic Similarity Demo")
    print("=" * 50)
    
    # Sample texts for similarity analysis
    texts: List[str] = [
        "What is the meaning of life?",
        "What is the purpose of existence?",
        "How do I bake a cake?",
        "What's the recipe for chocolate cake?",
        "Why do we exist in this universe?",
        "How to prepare delicious pastries?",
    ]
    
    print(f"Sample texts ({len(texts)} total):")
    for i, text in enumerate(texts, 1):
        print(f"  [{i}] {text}")
    
    try:
        # Configure API client
        configure_gemini_client()
        
        # Get embeddings
        embeddings: np.ndarray = get_embeddings(texts)
        
        # Compute similarity matrix
        similarity_matrix: np.ndarray = compute_similarity_matrix(embeddings)
        
        # Display results
        display_similarity_results(texts, similarity_matrix)
        
        print("Demo completed successfully! 🎉")
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
