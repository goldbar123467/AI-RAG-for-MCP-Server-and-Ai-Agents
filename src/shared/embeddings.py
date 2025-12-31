"""Ollama embedding client."""

from typing import List, Optional
import httpx

from .config import settings


class EmbeddingError(Exception):
    """Error generating embeddings."""
    pass


async def get_embedding(text: str, model: Optional[str] = None) -> List[float]:
    """
    Generate an embedding for the given text using Ollama.

    Args:
        text: The text to embed
        model: Optional model name, defaults to nomic-embed-text

    Returns:
        768-dimensional embedding vector

    Raises:
        EmbeddingError: If embedding generation fails
    """
    model = model or settings.embedding_model

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                f"{settings.ollama_url}/api/embeddings",
                json={
                    "model": model,
                    "prompt": text,
                }
            )
            response.raise_for_status()
            data = response.json()

            embedding = data.get("embedding")
            if not embedding:
                raise EmbeddingError(f"No embedding in response: {data}")

            if len(embedding) != settings.embedding_dim:
                raise EmbeddingError(
                    f"Expected {settings.embedding_dim} dimensions, got {len(embedding)}"
                )

            return embedding

        except httpx.HTTPStatusError as e:
            raise EmbeddingError(f"Ollama API error: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise EmbeddingError(f"Ollama connection error: {e}") from e


async def get_embeddings_batch(texts: List[str], model: Optional[str] = None) -> List[List[float]]:
    """
    Generate embeddings for multiple texts.

    Args:
        texts: List of texts to embed
        model: Optional model name

    Returns:
        List of embedding vectors
    """
    # Ollama doesn't support batch embeddings natively, so we do them sequentially
    # Could be parallelized with asyncio.gather if needed
    embeddings = []
    for text in texts:
        embedding = await get_embedding(text, model)
        embeddings.append(embedding)
    return embeddings
