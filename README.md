# DeepCite 📚🔍  
**AI-Powered Citation Recommendation System**  

DeepCite is a smart **citation recommendation system** that helps researchers find **relevant academic papers** based on the **title of their research**. It leverages **CrossRef, BM25, SPECTER embeddings, and citation analysis** to rank papers by **relevance, citation influence, and recency**, providing **titles, DOI links, and scores** for easy reference.  

## 🚀 Features  
- 📖 **Input**: Enter a **research paper title** as a query.  
- 🔍 **Retrieval**: Fetches relevant papers using **CrossRef/SemanticScholar API**.  
- 📊 **Ranking**: Uses a combination of:
  - **Relevance Score** (BM25, SPECTER)  
  - **Citation Score** (CrossRef citation count, citation graph analysis)  
  - **Recency Score** (Publication year boost)  
- 📜 **Output**: Returns **Top-N recommended papers** with:
  - **Title**  
  - **DOI Link**  
  - **Aggregated Score (Relevance + Citation Influence + Recency)**  

## 🏗️ Architecture  
1. **Query Processing**: User enters a research paper title.  
2. **Paper Retrieval**: Fetch relevant papers using **CrossRef/SemanticScholar API**.  
3. **Ranking Model**: Computes a final score using BM25, SPECTER, and citation-based ranking.  
4. **Output Formatting**: Returns a list of **Top-N recommended papers** with proper formatting.  

## 🔧 Tech Stack  
- **APIs**: [CrossRef API](https://www.crossref.org/), [Semantic Scholar API (optional)](https://www.semanticscholar.org/)  
- **Text Similarity**: BM25, SPECTER embeddings  
- **Ranking Algorithms**: Citation graph analysis, Recency scoring  

## 📦 Installation & Usage  
```bash
# Clone the repository
git clone https://github.com/yourusername/DeepCite.git  
cd DeepCite  

# Create a virtual environment  
python -m venv venv  
source venv/bin/activate  # On Windows use: venv\Scripts\activate  

# Install dependencies  
pip install -r requirements.txt  
