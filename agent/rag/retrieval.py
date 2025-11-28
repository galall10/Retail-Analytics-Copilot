"""
Document retrieval using TF-IDF + BM25 hybrid approach.
Chunks markdown documents and provides semantic search.
"""

import os
import re
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False


@dataclass
class Chunk:
    """A single document chunk with metadata."""
    id: str
    content: str
    source: str
    score: float = 0.0


class DocumentRetriever:
    """Retrieves relevant document chunks using TF-IDF + optional BM25."""

    def __init__(self, docs_dir: str = "docs", use_bm25: bool = True):
        """
        Initialize retriever and load all documents.
        
        Args:
            docs_dir: Directory containing markdown files
            use_bm25: If True, use BM25 (if available); else TF-IDF only
        """
        self.docs_dir = Path(docs_dir)
        self.chunks = []
        self.use_bm25 = use_bm25 and HAS_BM25
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.bm25 = None
        
        self._load_documents()

    def _load_documents(self):
        """Load and chunk all markdown documents."""
        if not self.docs_dir.exists():
            print(f"Warning: docs directory {self.docs_dir} not found")
            return

        for doc_path in sorted(self.docs_dir.glob("*.md")):
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Split by ## (level-2 headers) for chunking
            sections = re.split(r"\n## ", content)
            
            # First chunk includes title (if it starts with #)
            if sections[0].startswith("# "):
                first_chunk = sections[0]
                other_sections = sections[1:]
            else:
                first_chunk = None
                other_sections = sections
            
            # Process first chunk
            if first_chunk:
                chunk_id = f"{doc_path.stem}::chunk0"
                self.chunks.append(Chunk(
                    id=chunk_id,
                    content=first_chunk.strip(),
                    source=doc_path.stem
                ))
            
            # Process remaining sections
            for idx, section in enumerate(other_sections, start=1 if first_chunk else 0):
                chunk_id = f"{doc_path.stem}::chunk{idx}"
                self.chunks.append(Chunk(
                    id=chunk_id,
                    content=("## " + section).strip(),
                    source=doc_path.stem
                ))
        
        # Build vectorizers
        if self.chunks:
            self._build_vectorizers()

    def _build_vectorizers(self):
        """Build TF-IDF and BM25 indexes."""
        texts = [chunk.content for chunk in self.chunks]
        
        # TF-IDF
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=500,
            stop_words="english",
            lowercase=True,
            min_df=1
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        
        # BM25 (if available)
        if self.use_bm25:
            tokenized = [text.lower().split() for text in texts]
            self.bm25 = BM25Okapi(tokenized)

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        """
        Retrieve top-k most relevant chunks.
        
        Args:
            query: Natural language query
            top_k: Number of chunks to return
        
        Returns:
            List of Chunk objects with scores, sorted by relevance
        """
        if not self.chunks:
            return []
        
        top_k = min(top_k, len(self.chunks))
        
        if self.use_bm25:
            return self._retrieve_hybrid(query, top_k)
        else:
            return self._retrieve_tfidf(query, top_k)

    def _retrieve_tfidf(self, query: str, top_k: int) -> list[Chunk]:
        """Retrieve using TF-IDF only."""
        query_vec = self.tfidf_vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(scores[idx])
            results.append(chunk)
        
        return results

    def _retrieve_hybrid(self, query: str, top_k: int) -> list[Chunk]:
        """Retrieve using hybrid TF-IDF + BM25."""
        # TF-IDF scores
        query_vec = self.tfidf_vectorizer.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        
        # BM25 scores
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize and combine (equal weight)
        tfidf_norm = tfidf_scores / (np.max(tfidf_scores) + 1e-9)
        bm25_norm = bm25_scores / (np.max(bm25_scores) + 1e-9)
        combined_scores = 0.5 * tfidf_norm + 0.5 * bm25_norm
        
        # Get top-k indices
        top_indices = np.argsort(combined_scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            chunk = self.chunks[idx]
            chunk.score = float(combined_scores[idx])
            results.append(chunk)
        
        return results

    def get_all_chunks(self) -> list[Chunk]:
        """Return all chunks (for debugging/analysis)."""
        return self.chunks

    def search_by_source(self, source: str) -> list[Chunk]:
        """Get all chunks from a specific document source."""
        return [c for c in self.chunks if c.source == source]
