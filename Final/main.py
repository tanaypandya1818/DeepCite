import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
import time

# Updated API Endpoint for Semantic Scholar
SEMANTIC_SCHOLAR_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
HEADERS = {
    "User-Agent": "DeepCite/1.0",
}

# Function to fetch papers from Semantic Scholar API based on a query
def fetch_papers_by_keywords(keywords, fields="title,abstract,url,year,citationCount,authors", limit=1000):
    papers = []
    offset = 0
    batch_size = 100  # Smaller batch size to avoid rate limiting

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

                # Add delay between requests to avoid rate limiting
                time.sleep(1)

            elif response.status_code == 429:
                print(f"Rate limit exceeded. Waiting 30 seconds before retry...")
                time.sleep(30)  # Wait longer if rate limited
                continue
            else:
                print(f"Failed to fetch data, status code: {response.status_code}")
                break

        except Exception as e:
            print(f"Error fetching papers: {str(e)}")
            break

        # Check if we've reached all available papers
        if len(papers) >= offset:
            break

    return papers[:limit]

# Initialize the SentenceTransformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def get_embeddings(papers):
    """Generate embeddings for the list of papers using SentenceTransformer."""
    if not papers:
        return None

    titles_and_abstracts = []
    for paper in papers:
        title = paper.get('title', '')  # Default to empty string if title is missing
        abstract = paper.get('abstract', '')  # Default to empty string if abstract is missing
        title = title if title else ''  # Ensure title is a string
        abstract = abstract if abstract else ''  # Ensure abstract is a string
        titles_and_abstracts.append(title + " " + abstract)

    embeddings = model.encode(titles_and_abstracts, convert_to_tensor=True)
    return embeddings

# Function to create and store embeddings in FAISS index
def create_faiss_index(embeddings):
    """Create and store embeddings in FAISS index."""
    if embeddings is None:
        return None

    # Convert embeddings to a NumPy array for FAISS
    embeddings_np = np.array(embeddings.cpu().detach().numpy()).astype('float32')

    # Create the FAISS index (using L2 distance, Euclidean)
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)  # Add embeddings to the index
    return index

def get_bm25_scores(query, papers):
    """Compute BM25 similarity scores between query and papers."""
    if not papers:
        return []

    # Preprocess documents: tokenize title + abstract
    tokenized_corpus = [
        ((paper.get('title') or '') + ' ' + (paper.get('abstract') or '')).lower().split()
        for paper in papers]
    bm25 = BM25Okapi(tokenized_corpus)

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)

    return scores


def search_query(query, faiss_index, papers, top_k=1000):
    """Search query in FAISS index and retrieve semantic + BM25 scores."""
    if not papers or faiss_index is None:
        return [], [], []

    # Limit top_k to the number of available papers
    top_k = min(top_k, len(papers))
    if top_k == 0:
        return [], [], []

    # FAISS
    query_embedding = model.encode([query], convert_to_tensor=True)
    query_embedding_np = np.array(query_embedding.cpu().detach().numpy()).astype('float32')
    distances, indices = faiss_index.search(query_embedding_np, top_k)

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

def rank_by_weighted_score_hybrid(papers, faiss_dists, bm25_scores, weights=(0.5, 0.3, 0.2), hybrid_weights=(0.7, 0.3)):
    """Rank papers with hybrid relevance (FAISS + BM25) and citation/recency."""
    relevance_weight, citation_weight, recency_weight = weights
    w_faiss, w_bm25 = hybrid_weights

    # Normalize FAISS distances into similarity
    max_dist = max(faiss_dists) + 1e-5
    faiss_sims = [1 - (d / max_dist) for d in faiss_dists]

    # Normalize BM25 scores
    max_bm25 = max(bm25_scores) + 1e-5
    bm25_sims = [s / max_bm25 for s in bm25_scores]

    # Filter valid citation counts and years
    valid_citations = [p['citationCount'] for p in papers if p.get('citationCount') is not None]
    valid_years = [p['year'] for p in papers if p.get('year') is not None]

    if not valid_citations or not valid_years:
        raise ValueError("Missing valid 'citationCount' or 'year' values in papers.")

    max_citation = max(valid_citations) + 1e-5
    max_year = max(valid_years) + 1e-5
    min_year = min(valid_years)

    ranked = []
    for i, paper in enumerate(papers):
        combined_relevance = (w_faiss * faiss_sims[i]) + (w_bm25 * bm25_sims[i])

        citation = paper.get('citationCount')
        year = paper.get('year')

        norm_citation = (citation / max_citation) if citation is not None else 0
        norm_recency = ((year - min_year) / (max_year - min_year)) if year is not None else 0

        final_score = (
            relevance_weight * combined_relevance +
            citation_weight * norm_citation +
            recency_weight * norm_recency
        )

        ranked.append({
            "Title": paper['title'],
            "Abstract": paper.get('abstract', 'N/A'),
            "DOI": paper.get('url', "N/A"),
            "Citation Count": paper['citationCount'],
            "Year": paper['year'],
            "FAISS Similarity": round(faiss_sims[i], 4),
            "BM25 Similarity": round(bm25_sims[i], 4),
            "Combined Relevance": round(combined_relevance, 4),
            "Final Score": round(final_score, 4)
        })

    ranked.sort(key=lambda x: -x['Final Score'])
    return ranked

# Main function to fetch, rank, and return relevant papers based on a query
def get_ranked_papers(query, weights=(0.5, 0.3, 0.2), hybrid_weights=(0.7, 0.3), limit=1000):
    print(f"Searching for papers on: '{query}'")
    papers = fetch_papers_by_keywords(query, limit=limit)

    if not papers:
        return "No relevant papers found."

    print(f"Found relevant papers. Creating embeddings...")

    embeddings = get_embeddings(papers)
    faiss_index = create_faiss_index(embeddings)

    print("Searching and ranking papers...")
    recommended_papers, faiss_dists, bm25_scores = search_query(query, faiss_index, papers, top_k=len(papers))

    if not recommended_papers:
        return "No relevant papers could be ranked."

    ranked_papers = rank_by_weighted_score_hybrid(recommended_papers, faiss_dists, bm25_scores, weights, hybrid_weights)
    return ranked_papers

# Paper Display Class
class PaperDisplay:
    def __init__(self, papers, papers_per_page=10):
        self.papers = papers
        self.current_page = 0
        self.papers_per_page = papers_per_page
        self.total_pages = (len(papers) + self.papers_per_page - 1) // self.papers_per_page
    
    def get_page_papers(self):
        """Get papers for the current page"""
        start_idx = self.current_page * self.papers_per_page
        end_idx = min(start_idx + self.papers_per_page, len(self.papers))
        return self.papers[start_idx:end_idx], start_idx, end_idx
    
    def next_page(self):
        """Move to the next page if available"""
        if self.current_page < self.total_pages - 1:
            self.current_page += 1
            return True
        return False
    
    def prev_page(self):
        """Move to the previous page if available"""
        if self.current_page > 0:
            self.current_page -= 1
            return True
        return False