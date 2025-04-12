#!/usr/bin/env python3
"""
DeepCite - Academic Paper Search and Citation Tool

A command-line interface for searching and ranking academic papers
using semantic similarity and citation metrics.
"""

import argparse
import os
import sys
import time
import json
import requests
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama for cross-platform colored terminal text
init(autoreset=True)

# Check if required packages are installed
try:
    from sentence_transformers import SentenceTransformer
    from rank_bm25 import BM25Okapi
    import faiss
except ImportError:
    print(f"{Fore.RED}Error: Required packages are not installed.")
    print(f"{Fore.YELLOW}Please run: pip install -r requirements.txt")
    sys.exit(1)

# Configuration
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search/bulk"
HEADERS = {"User-Agent": "DeepCite/1.0"}

# Weights for ranking
DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)  # relevance, citations, recency
DEFAULT_HYBRID_WEIGHTS = (0.7, 0.3)  # semantic, lexical

# Global model to avoid reloading
global_model = None


def setup_model() -> SentenceTransformer:
    """Initialize and return the SentenceTransformer model."""
    global global_model
    if global_model is None:
        print(f"{Fore.BLUE}Loading language model... {Style.RESET_ALL}")
        global_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    return global_model


def fetch_papers_by_keywords(keywords: str, fields: str = "title,abstract,url,year,citationCount,authors", limit: int = 100) -> List[Dict[str, Any]]:
    """
    Fetch papers from Semantic Scholar API based on keywords.
    
    Args:
        keywords: Search query string
        fields: Fields to retrieve
        limit: Maximum number of papers to retrieve
        
    Returns:
        List of paper dictionaries
    """
    papers = []
    offset = 0
    batch_size = 25  # Smaller batch size to avoid rate limiting
    

        while len(papers) < limit:
            params = {
                "query": keywords,
                "fields": fields,
                "limit": min(batch_size, limit - len(papers)),
                "offset": offset
            }
            
            try:
                response = requests.get(SEMANTIC_SCHOLAR_BASE_URL, headers=HEADERS, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    new_papers = data.get("data", [])
                    
                    # Break if no new papers or fewer papers than requested (end of results)
                    if not new_papers:
                        break
                        
                    papers.extend(new_papers)
                    offset += len(new_papers)
                    pbar.update(len(new_papers))
                    
                    # Add delay between requests to avoid rate limiting
                    time.sleep(1)
                    
                elif response.status_code == 429:
                    print(f"\n{Fore.YELLOW}Rate limit exceeded. Waiting 30 seconds before retry...{Style.RESET_ALL}")
                    time.sleep(30)  # Wait longer if rate limited
                    continue
                else:
                    print(f"\n{Fore.RED}Failed to fetch data, status code: {response.status_code}{Style.RESET_ALL}")
                    break
                    
            except Exception as e:
                print(f"\n{Fore.RED}Error fetching papers: {str(e)}{Style.RESET_ALL}")
                break
                
            # Check if we've reached all available papers
            if len(papers) >= offset:
                break
                
    return papers[:limit]


def get_embeddings(papers: List[Dict[str, Any]], model: SentenceTransformer) -> np.ndarray:
    """
    Generate embeddings for the list of papers using SentenceTransformer.
    
    Args:
        papers: List of paper dictionaries
        model: Loaded SentenceTransformer model
        
    Returns:
        Array of embeddings
    """
    if not papers:
        return np.array([])
        
    titles_and_abstracts = []
    for paper in papers:
        title = paper.get('title', '')
        abstract = paper.get('abstract', '')
        title = title if title else ''
        abstract = abstract if abstract else ''
        titles_and_abstracts.append(title + " " + abstract)
        
    with tqdm(total=len(papers), desc="Creating embeddings", unit="paper") as pbar:
        embeddings = model.encode(titles_and_abstracts, show_progress_bar=False)
        pbar.update(len(papers))
        
    return embeddings


def create_faiss_index(embeddings: np.ndarray) -> Optional[faiss.Index]:
    """
    Create and store embeddings in FAISS index.
    
    Args:
        embeddings: Matrix of paper embeddings
        
    Returns:
        FAISS index object
    """
    if embeddings.size == 0:
        return None
        
    # Create the FAISS index (using L2 distance, Euclidean)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)  # Add embeddings to the index
    return index


def get_bm25_scores(query: str, papers: List[Dict[str, Any]]) -> np.ndarray:
    """
    Compute BM25 similarity scores between query and papers.
    
    Args:
        query: Search query string
        papers: List of paper dictionaries
        
    Returns:
        Array of BM25 scores
    """
    if not papers:
        return np.array([])
        
    # Preprocess documents: tokenize title + abstract
    tokenized_corpus = [
        ((paper.get('title') or '') + ' ' + (paper.get('abstract') or '')).lower().split()
        for paper in papers]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    return scores


def search_query(query: str, faiss_index: faiss.Index, papers: List[Dict[str, Any]], top_k: int = 20) -> Tuple[List[Dict[str, Any]], List[float], np.ndarray]:
    """
    Search query in FAISS index and retrieve semantic + BM25 scores.
    
    Args:
        query: Search query string
        faiss_index: FAISS index object
        papers: List of paper dictionaries
        top_k: Number of top papers to retrieve
        
    Returns:
        Tuple of (recommended papers, FAISS distances, BM25 scores)
    """
    if not papers or faiss_index is None:
        return [], [], np.array([])
        
    # Limit top_k to the number of available papers
    top_k = min(top_k, len(papers))
    if top_k == 0:
        return [], [], np.array([])
        
    # FAISS
    model = setup_model()
    query_embedding = model.encode([query])
    distances, indices = faiss_index.search(query_embedding, top_k)

    # Make sure indices are unique to avoid duplicates
    unique_indices = []
    seen = set()
    for idx in indices[0]:
        if idx not in seen and idx < len(papers):
            seen.add(idx)
            unique_indices.append(idx)
    
    recommended_papers = [papers[i] for i in unique_indices]
    
    # Get corresponding distances for the unique indices
    faiss_dists = [distances[0][list(indices[0]).index(i)] for i in unique_indices]

    # BM25
    bm25_scores = get_bm25_scores(query, recommended_papers)

    return recommended_papers, faiss_dists, bm25_scores


def rank_by_weighted_score_hybrid(papers: List[Dict[str, Any]], faiss_dists: List[float], bm25_scores: np.ndarray, 
                                 weights: Tuple[float, float, float] = DEFAULT_WEIGHTS, 
                                 hybrid_weights: Tuple[float, float] = DEFAULT_HYBRID_WEIGHTS) -> List[Dict[str, Any]]:
    """
    Rank papers with hybrid relevance (FAISS + BM25) and citation/recency.
    
    Args:
        papers: List of paper dictionaries
        faiss_dists: List of FAISS distances
        bm25_scores: Array of BM25 scores
        weights: Tuple of (relevance_weight, citation_weight, recency_weight)
        hybrid_weights: Tuple of (semantic_weight, lexical_weight)
        
    Returns:
        List of ranked paper dictionaries with scores
    """
    if not papers:
        return []
        
    relevance_weight, citation_weight, recency_weight = weights
    w_faiss, w_bm25 = hybrid_weights

    # Normalize FAISS distances into similarity
    max_dist = max(faiss_dists) if faiss_dists else 1
    max_dist = max_dist + 1e-5  # Avoid division by zero
    faiss_sims = [1 - (d / max_dist) for d in faiss_dists]

    # Normalize BM25 scores
    if isinstance(bm25_scores, np.ndarray):
        max_bm25 = np.max(bm25_scores) if bm25_scores.size > 0 else 1
        max_bm25 = max_bm25 + 1e-5  # Avoid division by zero
        bm25_sims = [float(s) / max_bm25 for s in bm25_scores]
    else:
        max_bm25 = max(bm25_scores) if bm25_scores else 1
        max_bm25 = max_bm25 + 1e-5  # Avoid division by zero
        bm25_sims = [s / max_bm25 for s in bm25_scores]
    
    # Fill in any missing scores if lengths don't match
    while len(faiss_sims) < len(papers):
        faiss_sims.append(0)
    while len(bm25_sims) < len(papers):
        bm25_sims.append(0)

    # Find max values for normalization, with safeguards
    max_citation = max((p.get('citationCount', 0) for p in papers), default=1) + 1e-5
    
    # Extract years with default for missing values
    years = [p.get('year', 2000) for p in papers]
    years = [y for y in years if y is not None]  # Filter out None values
    
    if not years:  # If all years are None
        max_year = 2023
        min_year = 2000
    else:
        max_year = max(years) + 1e-5
        min_year = min(years)
    
    # Ensure we don't divide by zero
    year_range = max(max_year - min_year, 1e-5)

    # Track paper IDs to avoid duplicates
    seen_papers = set()
    ranked = []
    
    for i, paper in enumerate(papers):
        # Skip duplicates based on paper ID or URL
        paper_id = paper.get('url', '') or paper.get('title', '')
        if paper_id in seen_papers:
            continue
        seen_papers.add(paper_id)
        
        # Get values with defaults
        citation_count = paper.get('citationCount', 0) or 0
        year = paper.get('year', min_year) or min_year
        
        # Ensure index is in range
        idx = min(i, len(faiss_sims)-1)
        
        combined_relevance = (w_faiss * faiss_sims[idx]) + (w_bm25 * bm25_sims[idx])
        norm_citation = citation_count / max_citation
        norm_recency = (year - min_year) / year_range

        final_score = (
            relevance_weight * combined_relevance +
            citation_weight * norm_citation +
            recency_weight * norm_recency
        )

        # Process authors with better error handling
        authors_list = paper.get('authors', [])
        if authors_list:
            try:
                authors_str = ', '.join([a.get('name', '') for a in authors_list[:3] if a.get('name')]) 
                if len(authors_list) > 3:
                    authors_str += '...'
            except Exception:
                authors_str = 'Authors not available'
        else:
            authors_str = 'Authors not available'

        ranked.append({
            "Title": paper.get('title', 'Title not available'),
            "DOI": paper.get('url', "N/A"),
            "Citation Count": citation_count,
            "Year": year,
            "FAISS Similarity": round(faiss_sims[idx], 4),
            "BM25 Similarity": round(bm25_sims[idx], 4),
            "Combined Relevance": round(combined_relevance, 4),
            "Final Score": round(final_score, 4),
            "Authors": authors_str,
            "Abstract": paper.get('abstract', 'No abstract available') if paper.get('abstract') is not None else 'No abstract available'
        })

    # Sort by final score and ensure uniqueness
    ranked.sort(key=lambda x: -x['Final Score'])
    
    # Remove any potential duplicates that might have slipped through
    unique_ranked = []
    seen_titles = set()
    
    for paper in ranked:
        if paper['Title'] not in seen_titles:
            seen_titles.add(paper['Title'])
            unique_ranked.append(paper)
    
    return unique_ranked


def get_ranked_papers(query: str, weights: Tuple[float, float, float] = DEFAULT_WEIGHTS, 
                     hybrid_weights: Tuple[float, float] = DEFAULT_HYBRID_WEIGHTS, 
                     limit: int = 1000) -> List[Dict[str, Any]]:
    """
    Main function to fetch, rank, and return relevant papers based on a query.
    
    Args:
        query: Search query string
        weights: Tuple of (relevance_weight, citation_weight, recency_weight)
        hybrid_weights: Tuple of (semantic_weight, lexical_weight)
        limit: Maximum number of papers to fetch
        
    Returns:
        List of ranked paper dictionaries
    """
    print(f"{Fore.GREEN}Searching for papers on: '{query}'{Style.RESET_ALL}")
    papers = fetch_papers_by_keywords(query, limit=limit)
    
    if not papers:
        return []
    
    print(f"{Fore.BLUE}Found {len(papers)} papers.{Style.RESET_ALL}")
    
    model = setup_model()
    embeddings = get_embeddings(papers, model)
    faiss_index = create_faiss_index(embeddings)
    
    print(f"{Fore.BLUE}Searching and ranking papers...{Style.RESET_ALL}")
    recommended_papers, faiss_dists, bm25_scores = search_query(query, faiss_index, papers, top_k=min(20, len(papers)))
    
    if not recommended_papers:
        return []
    
    ranked_papers = rank_by_weighted_score_hybrid(recommended_papers, faiss_dists, bm25_scores, weights, hybrid_weights)
    return ranked_papers


def display_paper(paper: Dict[str, Any], idx: int = None, detailed: bool = False) -> None:
    """
    Display a single paper's information in a formatted way.
    
    Args:
        paper: Paper dictionary
        idx: Optional index for numbering
        detailed: Whether to show detailed information
    """
    title_display = f"{idx}. " if idx is not None else ""
    title_display += f"{Fore.CYAN}{paper.get('Title', 'Title not available')}{Style.RESET_ALL}"
    
    print(title_display)
    
    # Display authors with fallback message
    authors = paper.get('Authors', '')
    print(f"   {Fore.BLUE}Authors:{Style.RESET_ALL} {authors if authors else 'Authors not available'}")
    
    # Display year and citations with fallback messages
    year = paper.get('Year', 'N/A')
    citations = paper.get('Citation Count', 'N/A')
    print(f"   {Fore.BLUE}Year:{Style.RESET_ALL} {year} | {Fore.BLUE}Citations:{Style.RESET_ALL} {citations}")
    
    # Display DOI with fallback message
    doi = paper.get('DOI', '')
    print(f"   {Fore.BLUE}DOI:{Style.RESET_ALL} {doi if doi and doi != 'N/A' else 'DOI not available'}")
    
    # Display relevance score
    final_score = paper.get('Final Score', 0)
    print(f"   {Fore.YELLOW}Relevance Score:{Style.RESET_ALL} {round(final_score * 100, 2)}%")
    
    # Display abstract if detailed view is requested
    if detailed:
        print()
        print(f"   {Fore.BLUE}Abstract:{Style.RESET_ALL}")
        abstract = paper.get('Abstract', '')
        if abstract and abstract != 'No abstract available':
            # Print wrapped abstract
            for i in range(0, len(abstract), 80):
                print(f"   {abstract[i:i+80]}")
        else:
            print("   Abstract not available")
    
    print()



def display_papers(papers: List[Dict[str, Any]], count: int = 10, detailed: bool = False) -> None:
    """
    Display multiple papers in a formatted list.
    
    Args:
        papers: List of paper dictionaries
        count: Number of papers to display
        detailed: Whether to show detailed information
    """
    if not papers:
        print(f"{Fore.RED}No relevant papers found.{Style.RESET_ALL}")
        return
    
    print(f"\n{Fore.GREEN}Top {min(count, len(papers))} Recommended Papers:{Style.RESET_ALL}\n")
    
    for i, paper in enumerate(papers[:count], 1):
        display_paper(paper, i, detailed)


def save_results(papers: List[Dict[str, Any]], filename: str) -> None:
    """
    Save search results to a JSON file.
    
    Args:
        papers: List of paper dictionaries
        filename: Output filename
    """
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(papers, f, indent=2)
        print(f"{Fore.GREEN}Results saved to {filename}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving results: {str(e)}{Style.RESET_ALL}")


def interactive_mode() -> None:
    """Run the CLI in interactive mode."""
    print(f"{Fore.GREEN}=== DeepCite - Academic Paper Search and Citation Tool ==={Style.RESET_ALL}")
    print(f"{Fore.BLUE}Type 'exit' or 'quit' to leave the program at any time.{Style.RESET_ALL}")
    print(f"{Fore.BLUE}Press Escape or Ctrl+C to exit the program.{Style.RESET_ALL}")
    
    # Load model once at the beginning
    setup_model()
    
    while True:
        print("\n" + "=" * 60)
        query = input(f"{Fore.GREEN}Enter search query: {Style.RESET_ALL}")
        
        if query.lower() in ['exit', 'quit']:
            print(f"{Fore.YELLOW}Exiting DeepCite. Goodbye!{Style.RESET_ALL}")
            break
            
        if not query.strip():
            print(f"{Fore.RED}Please enter a valid search query.{Style.RESET_ALL}")
            continue
            
        # Set default papers to search as 1000
        print(f"{Fore.BLUE}Searching for papers...{Style.RESET_ALL}")
        limit = 1000
        
        # Always show detailed information by default
        detailed = True
        
        # Search for papers
        ranked_papers = get_ranked_papers(query, limit=limit)
        
        if ranked_papers:
            # Display top 10 results by default with details
            display_count = 10
            print(f"\n{Fore.GREEN}Top {min(display_count, len(ranked_papers))} Recommended Papers:{Style.RESET_ALL}\n")
            
            # Display first 10 papers with details
            for i, paper in enumerate(ranked_papers[:display_count], 1):
                display_paper(paper, i, detailed=True)
            
            # Show more papers if requested
            displayed_so_far = display_count
            while displayed_so_far < len(ranked_papers):
                more_input = input(f"{Fore.GREEN}Show more papers? (y/n, default: n): {Style.RESET_ALL}")
                if more_input.lower() != 'y':
                    break
                    
                # Show the next 10 papers (or fewer if we're near the end)
                next_batch = min(10, len(ranked_papers) - displayed_so_far)
                print(f"\n{Fore.GREEN}Showing papers {displayed_so_far+1} to {displayed_so_far+next_batch}:{Style.RESET_ALL}\n")
                
                for i in range(displayed_so_far, displayed_so_far + next_batch):
                    display_paper(ranked_papers[i], i+1, detailed=True)
                
                displayed_so_far += next_batch
            
            # Ask if user wants to save results
            save_input = input(f"{Fore.GREEN}Save results to file? (y/n, default: n): {Style.RESET_ALL}")
            if save_input.lower() == 'y':
                filename = input(f"{Fore.GREEN}Enter filename (default: results.json): {Style.RESET_ALL}")
                filename = filename.strip() if filename.strip() else "results.json"
                save_results(ranked_papers, filename)
                
            # Ask if user wants to view a specific paper in detail
            while True:
                detail_input = input(f"{Fore.GREEN}View specific paper by number (or press Enter to continue): {Style.RESET_ALL}")
                if not detail_input.strip():
                    break
                    
                try:
                    paper_idx = int(detail_input) - 1
                    if 0 <= paper_idx < len(ranked_papers):
                        print("\n" + "=" * 60)
                        display_paper(ranked_papers[paper_idx], paper_idx + 1, True)
                    else:
                        print(f"{Fore.RED}Invalid paper number.{Style.RESET_ALL}")
                except ValueError:
                    print(f"{Fore.RED}Please enter a valid number.{Style.RESET_ALL}")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DeepCite - Academic Paper Search and Citation Tool")
    parser.add_argument("-q", "--query", type=str, help="Search query")
    parser.add_argument("-l", "--limit", type=int, default=50, help="Maximum number of papers to fetch (default: 50)")
    parser.add_argument("-c", "--count", type=int, default=10, help="Number of results to display (default: 10)")
    parser.add_argument("-d", "--detailed", action="store_true", help="Show detailed information including abstracts")
    parser.add_argument("-s", "--save", type=str, help="Save results to specified file")
    parser.add_argument("-i", "--interactive", action="store_true", help="Run in interactive mode")
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for the CLI."""
    args = parse_args()
    
    # Run in interactive mode if specified or if no query is provided
    if args.interactive or not args.query:
        interactive_mode()
        return
        
    # Run in command line mode
    ranked_papers = get_ranked_papers(args.query, limit=args.limit)
    
    if ranked_papers:
        display_papers(ranked_papers, args.count, args.detailed)
        
        if args.save:
            save_results(ranked_papers, args.save)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Program interrupted. Exiting.{Style.RESET_ALL}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}An error occurred: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)