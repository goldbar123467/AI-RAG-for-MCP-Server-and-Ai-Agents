"""Feature extraction for XGBoost quality prediction."""

import logging
import re
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import numpy as np


logger = logging.getLogger(__name__)


def parse_pgvector_string(s: str) -> List[float]:
    """
    Parse pgvector string representation to list of floats.

    pgvector returns embeddings as strings like "[0.123, 0.456, ...]" when
    accessed via raw SQL queries. This function parses that back to a list.

    Args:
        s: String representation from pgvector, e.g. "[0.1, 0.2, 0.3]"

    Returns:
        List of floats

    Raises:
        ValueError: If the string cannot be parsed as a valid vector
    """
    if not s:
        raise ValueError("Empty string cannot be parsed as pgvector")
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        raise ValueError(f"Invalid pgvector format: {s[:50]}...")
    inner = s[1:-1].strip()
    if not inner:
        return []
    try:
        return [float(x.strip()) for x in inner.split(',')]
    except ValueError as e:
        raise ValueError(f"Failed to parse pgvector elements: {e}") from e


# Reasoning words that indicate higher quality memories
REASONING_WORDS = {
    "because", "therefore", "thus", "hence", "since", "so",
    "consequently", "as a result", "due to", "leads to",
    "implies", "means that", "suggests", "indicates",
    "if", "then", "when", "unless", "although", "however",
    "but", "yet", "despite", "whereas", "while",
    "first", "second", "finally", "next", "additionally",
    "learned", "discovered", "realized", "found that",
    "caused", "fixed", "solved", "resolved", "prevented",
}

# Code indicators
CODE_PATTERNS = [
    r"```",  # Code blocks
    r"`[^`]+`",  # Inline code
    r"def\s+\w+\s*\(",  # Python functions
    r"function\s+\w+\s*\(",  # JS functions
    r"class\s+\w+",  # Class definitions
    r"\w+\.\w+\(",  # Method calls
    r"import\s+\w+",  # Imports
    r"from\s+\w+\s+import",  # Python imports
    r"=>\s*{",  # Arrow functions
    r"const\s+\w+\s*=",  # JS const
    r"let\s+\w+\s*=",  # JS let
    r"var\s+\w+\s*=",  # JS var
]

# Source trust levels (higher = more trusted)
SOURCE_TRUST = {
    "human": 1.0,
    "manual": 1.0,
    "user": 0.9,
    "verified": 0.9,
    "claude": 0.7,
    "agent": 0.5,
    "auto": 0.4,
    "inferred": 0.3,
    "unknown": 0.5,
}


@dataclass
class Features:
    """Extracted features for quality prediction."""

    # Text features
    length: int
    word_count: int
    unique_word_ratio: float
    sentence_count: int

    # Content indicators
    has_numbers: bool
    has_code: bool
    has_reasoning_words: bool
    reasoning_word_count: int

    # Source trust
    source_trust: float

    # Embedding statistics (if available)
    embedding_mean: Optional[float] = None
    embedding_std: Optional[float] = None
    embedding_min: Optional[float] = None
    embedding_max: Optional[float] = None

    def to_array(self) -> np.ndarray:
        """Convert features to numpy array for model input."""
        return np.array([
            self.length,
            self.word_count,
            self.unique_word_ratio,
            self.sentence_count,
            float(self.has_numbers),
            float(self.has_code),
            float(self.has_reasoning_words),
            self.reasoning_word_count,
            self.source_trust,
            self.embedding_mean or 0.0,
            self.embedding_std or 0.0,
            self.embedding_min or 0.0,
            self.embedding_max or 0.0,
        ], dtype=np.float32)

    @staticmethod
    def feature_names() -> List[str]:
        """Get feature names for model interpretability."""
        return [
            "length",
            "word_count",
            "unique_word_ratio",
            "sentence_count",
            "has_numbers",
            "has_code",
            "has_reasoning_words",
            "reasoning_word_count",
            "source_trust",
            "embedding_mean",
            "embedding_std",
            "embedding_min",
            "embedding_max",
        ]


def extract_features(
    content: str,
    source: Optional[str] = None,
    embedding: Optional[Union[List[float], str]] = None,
) -> Features:
    """
    Extract features from memory content for quality prediction.

    Args:
        content: The memory content text
        source: Optional source identifier for trust scoring
        embedding: Optional embedding vector for statistics

    Returns:
        Features dataclass with extracted values
    """
    # Basic text stats
    length = len(content)
    words = content.lower().split()
    word_count = len(words)
    unique_words = set(words)
    unique_word_ratio = len(unique_words) / max(word_count, 1)

    # Sentence count (simple heuristic)
    sentences = re.split(r"[.!?]+", content)
    sentence_count = len([s for s in sentences if s.strip()])

    # Number detection
    has_numbers = bool(re.search(r"\d+", content))

    # Code detection
    has_code = any(re.search(pattern, content) for pattern in CODE_PATTERNS)

    # Reasoning word detection
    content_lower = content.lower()
    reasoning_matches = [
        word for word in REASONING_WORDS
        if word in content_lower
    ]
    has_reasoning_words = len(reasoning_matches) > 0
    reasoning_word_count = len(reasoning_matches)

    # Source trust
    source_trust = SOURCE_TRUST.get(
        (source or "unknown").lower(),
        SOURCE_TRUST["unknown"]
    )

    # Embedding statistics
    embedding_mean = None
    embedding_std = None
    embedding_min = None
    embedding_max = None

    if embedding:
        # Handle pgvector string format from raw SQL queries
        if isinstance(embedding, str):
            try:
                embedding = parse_pgvector_string(embedding)
            except ValueError as e:
                logger.warning(f"Failed to parse embedding: {e}")
                embedding = None

        if embedding:
            emb_array = np.array(embedding)
            embedding_mean = float(np.mean(emb_array))
            embedding_std = float(np.std(emb_array))
            embedding_min = float(np.min(emb_array))
            embedding_max = float(np.max(emb_array))

    return Features(
        length=length,
        word_count=word_count,
        unique_word_ratio=unique_word_ratio,
        sentence_count=sentence_count,
        has_numbers=has_numbers,
        has_code=has_code,
        has_reasoning_words=has_reasoning_words,
        reasoning_word_count=reasoning_word_count,
        source_trust=source_trust,
        embedding_mean=embedding_mean,
        embedding_std=embedding_std,
        embedding_min=embedding_min,
        embedding_max=embedding_max,
    )


def heuristic_quality_score(features: Features) -> float:
    """
    Compute a heuristic quality score before ML model is trained.

    This provides a baseline until enough training data is collected.

    Args:
        features: Extracted features

    Returns:
        Quality score between 0 and 1
    """
    score = 0.5  # Base score

    # Length bonus (prefer substantial content, but not too long)
    if 100 <= features.length <= 2000:
        score += 0.1
    elif features.length > 2000:
        score += 0.05  # Still good, but might be verbose
    elif features.length < 10:
        score -= 0.3  # Trivially short, almost certainly useless
    elif features.length < 50:
        score -= 0.15  # Too short

    # Word diversity
    if features.unique_word_ratio > 0.6:
        score += 0.05

    # Code presence (usually specific and actionable)
    if features.has_code:
        score += 0.1

    # Reasoning words (indicates explanation/context)
    if features.has_reasoning_words:
        score += 0.05 * min(features.reasoning_word_count, 3)

    # Numbers (often specific/measurable)
    if features.has_numbers:
        score += 0.05

    # Source trust
    score += (features.source_trust - 0.5) * 0.2

    # Clamp to [0, 1]
    return max(0.0, min(1.0, score))
