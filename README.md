# DeepCite 📚🔍

<p align="center">
  <em>Literature recommendation system</em>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#tech-stack">Tech Stack</a> •
  <a href="#installation--usage">Installation</a> •
  <a href="#example">Example</a> •
  <a href="#contributing">Contributing</a> •
  <a href="#license">License</a>
</p>

## 📋 Overview

DeepCite is an intelligent citation recommendation engine designed to help researchers discover highly relevant academic papers based on their research topic or paper title. By leveraging state-of-the-art information retrieval techniques and citation analysis, DeepCite provides a curated list of the most important papers to cite in your research.

## ✨ Features

### Input
- **Research Query**: Enter your paper title or research topic as a query

### Retrieval & Processing
- **Multi-Source Data Integration**: Fetches candidate papers from Semantic Scholar APIs
- **Comprehensive Paper Metadata**: Collects titles, abstracts, authors, publication years, and citation networks
- **Advanced Text Processing**: Applies preprocessing to optimize matching quality

### Intelligent Ranking
DeepCite uses a sophisticated multi-factor ranking algorithm that considers:

- **Relevance Score (40%)**
  - BM25 text similarity between your query and paper titles/abstracts
  - SPECTER embeddings for semantic similarity capturing deeper conceptual relationships
  
- **Citation Score (40%)**
  - Raw citation counts from CrossRef
  - Citation influence analysis using PageRank-inspired algorithms
  - Reference network analysis to identify foundational papers in the field
  
- **Recency Score (20%)**
  - Publication year weighting to balance classic papers with recent developments
  - Temporal citation velocity to identify emerging influential work

### Output
DeepCite returns the top-ranked papers with:
- Paper title with formatting preserved
- Direct DOI link for easy access
- Publication year and journal/conference name
- Aggregated relevance score with component breakdown
- Brief explanation of why each paper was recommended

## 🏗️ Architecture

1. **Query Analyzer**
   - Processes user input to extract key research concepts
   - Expands queries with field-specific terminology

2. **Paper Retrieval Engine**
   - Primary API integration with CrossRef for broad coverage
   - Secondary integration with Semantic Scholar for enhanced metadata
   - Caching system for improved performance on common queries

3. **Citation Analysis Module**
   - Constructs citation graphs from retrieved papers
   - Applies network analysis algorithms to identify influential nodes
   - Calculates citation-based importance metrics

4. **Semantic Processing Pipeline**
   - Implements BM25 ranking for efficient lexical matching
   - Utilizes SPECTER embeddings for deep semantic understanding
   - Combines multiple similarity measures for robust matching

5. **Recommendation Generator**
   - Aggregates scores using configurable weighting schemes
   - Applies diversity algorithms to ensure varied recommendations
   - Formats output with clean, researcher-friendly presentation

## 🔧 Tech Stack

- **APIs**:
  - Semantic Scholar API 
  
- **Text Processing**:
  - BM25 ranking algorithm for efficient retrieval
  - SPECTER embeddings (SciBERT-based) for semantic similarity
  - NLTK and spaCy for text preprocessing
  
- **Ranking Algorithms**:
  - Custom citation graph analysis
  - Temporal weighting functions
  - Configurable multi-factor scoring
  
- **Backend**:
  - Python 3.13+

## 📦 Installation & Usage

### Prerequisites
- Python 3.13.1 or higher
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/devamjariwala24/DeepCite.git
cd DeepCite

# Optional: Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate


### Running DeepCite

```bash
# Start the application
python run.py  # Use python3 run.py on macOS/Linux if needed
```

### Using the CLI Interface

When you run the application, it will:
1. Check for necessary libraries and install any that are missing (this might take a moment)
2. Prompt you to enter your research query
3. Display the top 10 recommended papers

**Navigation Commands:**
- Press `p` to view previous results (if not on first page)
- Press `n` to view next page (if not on last page)
- Press `s` to search for a new query
- Press `q` to quit the program

## 📊 Example

**Input Query**: "Transformer models for natural language processing"

**Sample Output**:
```
DeepCite Results:
-----------------
1. "Attention Is All You Need" (2017)
   DOI: https://doi.org/10.48550/arXiv.1706.03762
   Score: 0.94 (Relevance: 0.91, Citation: 0.98, Recency: 0.85)
   Note: Foundational paper introducing the Transformer architecture

2. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" (2019)
   DOI: https://doi.org/10.48550/arXiv.1810.04805
   Score: 0.92 (Relevance: 0.95, Citation: 0.97, Recency: 0.87)
   Note: Key application of Transformers that revolutionized NLP benchmarks
   
[...additional results...]
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

Project Link: [https://github.com/devamjariwala24/DeepCite](https://github.com/devamjariwala24/DeepCite)


